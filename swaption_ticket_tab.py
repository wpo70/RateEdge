"""
swaption_ticket_tab.py
RateEdge Swaption Pricer — Trade Ticket Tab

Generates a swaption trade ticket from pricer output,
stores to Supabase, and produces the Citycom JSON payload
ready for MarkitWire submission via FIXEdge.

Wire into app_streamlit.py:
    from swaption_ticket_tab import render_ticket_tab
    with tab_ticket:
        render_ticket_tab(st.session_state)

Session state keys set by pricer when pricing runs:
    ticket_option_type      'Straddle'|'Payers'|'Receivers'
    ticket_option_expiry    '3m'  (label)
    ticket_option_expiry_y   0.25 (years)
    ticket_swap_term        '5Y'
    ticket_swap_term_y       5.0
    ticket_expiry_date      '2026-07-06'
    ticket_swap_start_date  '2026-10-07'
    ticket_strike_rate       4.25  (%)
    ticket_premium_bp        164.5 (bp)
    ticket_notional          100   (millions)
    ticket_currency          'AUD'
"""

import streamlit as st
import json
from datetime import datetime, date, timedelta
import os

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# ─── Supabase ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase():
    if not SUPABASE_AVAILABLE:
        return None
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
    key = st.secrets.get("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "")))
    if not url or not key:
        return None
    return create_client(url, key)

@st.cache_data(ttl=300, show_spinner=False)
def load_banks():
    sb = get_supabase()
    if not sb: return []
    return (sb.table("re_banks").select("*").order("bank").execute()).data or []

@st.cache_data(ttl=300, show_spinner=False)
def load_traders():
    sb = get_supabase()
    if not sb: return []
    return (sb.table("re_traders").select("*").order("lastname").execute()).data or []

@st.cache_data(ttl=300, show_spinner=False)
def load_divisions():
    sb = get_supabase()
    if not sb: return []
    return (sb.table("re_bank_divisions").select("*").execute()).data or []

@st.cache_data(ttl=300, show_spinner=False)
def load_bic():
    sb = get_supabase()
    if not sb: return []
    return (sb.table("re_bic").select("*").execute()).data or []

# ─── BIC 5-rule engine (port of bic.js getMatchingBic) ───────────────────────
def resolve_bic(buyer_bank_id, seller_bank_id, buyer_trader_id, seller_trader_id, all_bic):
    buyer_codes  = [b for b in all_bic if b["bank_id"] == buyer_bank_id]
    seller_codes = [b for b in all_bic if b["bank_id"] == seller_bank_id]

    bic_buyer = bic_seller = None

    def set_b(c):
        nonlocal bic_buyer
        if not bic_buyer: bic_buyer = c

    def set_s(c):
        nonlocal bic_seller
        if not bic_seller: bic_seller = c

    SPECIAL = {'BNPAFRPP','DEUTDEFF','DEUTGB2L','UBSWAU2SXXX'}

    def classify(codes):
        dcl, branch, special = None, [], []
        for c in codes:
            bic = c.get('markitbiccode','')
            if 'DCL' in bic:          dcl = c
            elif bic in SPECIAL:      special.append(c)
            else:                     branch.append(c)
        return dcl, branch, special

    # Rule 2 – Nomura (bank_id=3)
    NIP = {2,5,6,7,16,17,18,19,22,26}
    if buyer_bank_id == 3:
        dcl, branch, _ = classify(buyer_codes)
        c = next((x for x in branch if 'NOM' in x.get('markitbiccode','')),None) \
            if seller_bank_id in NIP else dcl
        set_b(c)
    elif seller_bank_id == 3:
        dcl, branch, _ = classify(seller_codes)
        c = next((x for x in branch if 'NOM' in x.get('markitbiccode','')),None) \
            if buyer_bank_id in NIP else dcl
        set_s(c)

    # Rule 3 – Deutsche (bank_id=25)
    DEUT_FF = {78,79}
    if buyer_bank_id == 25:
        _, _, special = classify(buyer_codes)
        pat = 'FF' if buyer_trader_id in DEUT_FF else 'GB2L'
        set_b(next((x for x in special if pat in x.get('markitbiccode','')),None))
    elif seller_bank_id == 25:
        _, _, special = classify(seller_codes)
        pat = 'FF' if seller_trader_id in DEUT_FF else 'GB2L'
        set_s(next((x for x in special if pat in x.get('markitbiccode','')),None))

    # Rule 4 – BNP(4)+UBS(14)
    if buyer_bank_id == 4 and seller_bank_id == 14:
        _, _, bs = classify(buyer_codes); _, _, ss = classify(seller_codes)
        set_b(next((x for x in bs if 'FRPP' in x.get('markitbiccode','')),None))
        set_s(next((x for x in ss if 'SXXX' in x.get('markitbiccode','')),None))
    elif seller_bank_id == 4 and buyer_bank_id == 14:
        _, _, bs = classify(buyer_codes); _, _, ss = classify(seller_codes)
        set_b(next((x for x in bs if 'SXXX' in x.get('markitbiccode','')),None))
        set_s(next((x for x in ss if 'FRPP' in x.get('markitbiccode','')),None))

    # Rule 5 – Default DCL → Branch[0]
    dcl, branch, _ = classify(buyer_codes);  set_b(dcl or (branch[0] if branch else None))
    dcl, branch, _ = classify(seller_codes); set_s(dcl or (branch[0] if branch else None))

    return bic_buyer, bic_seller

# ─── Citycom/FIXEdge JSON builder ────────────────────────────────────────────
def build_mw_json(t):
    """
    Builds JSON per Citycom swaption spec (trade_example_swaption.json)
    payer/receiver refer to the UNDERLYING SWAP fixed leg direction.
    """
    sfx = "_AU"
    b_code = t["buyer_ov_bank_id"]
    s_code = t["seller_ov_bank_id"]

    if t["option_type"] == "Payers":
        payer, receiver = f"{b_code}{sfx}", f"{s_code}{sfx}"
    elif t["option_type"] == "Receivers":
        payer, receiver = f"{s_code}{sfx}", f"{b_code}{sfx}"
    else:  # Straddle
        payer, receiver = f"{b_code}{sfx}", f"{s_code}{sfx}"

    basis = 0.25 if t["swap_term_y"] <= 3 else 0.5

    payload = {
        "event":       "option",
        "type":        "outright",
        "option_type": "swaption",
        "timestamp":   t["timestamp"],
        "leg1": {
            "payer":         payer,
            "receiver":      receiver,
            "volume":        t["notional"],
            "strike":        round(t["strike_rate"] / 100, 8),
            "premium":       t["premium_bp"],
            "tenor":         t["swap_term_y"],
            "option_expiry": t["option_expiry_y"],
            "basis":         basis,
            "currency":      t["currency"],
        }
    }
    if t["option_type"] == "Straddle":
        payload["leg1"]["straddle"] = True

    return payload

# ─── Save to Supabase ─────────────────────────────────────────────────────────
def save_to_supabase(t, mw_json):
    sb = get_supabase()
    if not sb:
        return False, "Supabase not configured"
    record = {
        "trade_id":        t["trade_id"],
        "timestamp":       t["timestamp"],
        "currency":        t["currency"],
        "option_type":     t["option_type"],
        "option_expiry":   t["option_expiry"],
        "swap_term":       t["swap_term"],
        "strike_rate":     t["strike_rate"],
        "premium_bp":      t["premium_bp"],
        "notional":        t["notional"],
        "expiry_date":     t["expiry_date"],
        "swap_start_date": t["swap_start_date"],
        "premium_date":    str(t["premium_date"]),
        "settlement":      t["settlement"],
        "spot_or_fwd":     t["spot_or_fwd"],
        "clearhouse":      t.get("clearhouse",""),
        "sef":             t.get("sef", False),
        "buyer_bank":      t["buyer_bank_name"],
        "buyer_trader":    t["buyer_trader_name"],
        "seller_bank":     t["seller_bank_name"],
        "seller_trader":   t["seller_trader_name"],
        "buyer_brokerage": t.get("buyer_brokerage"),
        "seller_brokerage":t.get("seller_brokerage"),
        "bic_buyer":       t.get("bic_buyer",""),
        "bic_seller":      t.get("bic_seller",""),
        "mw_payload":      json.dumps(mw_json),
        "mw_status":       "Pending",
        "source":          "pricer",
    }
    try:
        sb.table("swaption_orders").insert(record).execute()
        return True, None
    except Exception as e:
        return False, str(e)

# ─── Main ─────────────────────────────────────────────────────────────────────
def render_ticket_tab(ss):
    st.subheader("📋 Swaption Trade Ticket → MarkitWire")

    # Load data
    all_banks     = load_banks()
    all_traders   = load_traders()
    all_divisions = load_divisions()
    all_bic       = load_bic()

    if not all_banks:
        st.error("No counterparty data. Populate re_banks / re_traders / re_bank_divisions / re_bic in Supabase.")
        return

    # Pull from pricer session state
    option_type_pre = getattr(ss, "ticket_option_type",     "Straddle")
    option_expiry   = getattr(ss, "ticket_option_expiry",   "—")
    option_expiry_y = getattr(ss, "ticket_option_expiry_y",  0.0)
    swap_term       = getattr(ss, "ticket_swap_term",        "—")
    swap_term_y     = getattr(ss, "ticket_swap_term_y",       0.0)
    expiry_date     = getattr(ss, "ticket_expiry_date",      "")
    swap_start      = getattr(ss, "ticket_swap_start_date",  "")
    strike_rate     = getattr(ss, "ticket_strike_rate",       0.0)
    premium_bp      = getattr(ss, "ticket_premium_bp",        0.0)
    notional        = getattr(ss, "ticket_notional",          100)
    currency        = getattr(ss, "ticket_currency",          "AUD")
    pricing_ok      = bool(expiry_date and swap_start and premium_bp)

    # ── Pricing (editable) ────────────────────────────────────────────────────
    st.markdown("#### Pricing")
    if not pricing_ok:
        st.warning("Run the pricer to populate pricing fields.")

    def next_business_day(d):
        """Expiry + 1 mod following — skip weekends."""
        d = d + timedelta(days=1)
        if d.weekday() == 5: d += timedelta(days=2)   # Saturday → Monday
        if d.weekday() == 6: d += timedelta(days=1)   # Sunday → Monday
        return d

    def add_months(d, months):
        import calendar
        month = d.month - 1 + months
        year  = d.year + month // 12
        month = month % 12 + 1
        day   = min(d.day, calendar.monthrange(year, month)[1])
        return d.replace(year=year, month=month, day=day)

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(f"<div style='font-size:13px;color:#94a3b8;margin-bottom:4px'>Option Expiry: <b style='color:#e2e8f0'>{option_expiry}</b></div>", unsafe_allow_html=True)
        try:
            _exp_default = datetime.strptime(expiry_date, "%Y-%m-%d").date() if expiry_date else date.today()
        except:
            _exp_default = date.today()
        expiry_date_inp = st.date_input("Expiry Date", value=_exp_default, key="tix_expiry_date")
        expiry_date = expiry_date_inp.strftime("%Y-%m-%d")

    with p2:
        st.markdown(f"<div style='font-size:13px;color:#94a3b8;margin-bottom:4px'>Swap Term: <b style='color:#e2e8f0'>{swap_term}</b></div>", unsafe_allow_html=True)
        # Swap Start: expiry + 1 BD mod following
        _start_default = next_business_day(expiry_date_inp)
        try:
            # Use pricer value if it makes sense (after expiry)
            _ps = datetime.strptime(swap_start, "%Y-%m-%d").date() if swap_start else _start_default
            if _ps <= expiry_date_inp: _ps = _start_default
        except:
            _ps = _start_default
        swap_start_inp = st.date_input("Swap Start", value=_ps, key="tix_swap_start",
            help="Expiry + 1 business day (mod following)")
        swap_start = swap_start_inp.strftime("%Y-%m-%d")

        # Swap End
        _freq_m = 3 if swap_term_y <= 3 else 6
        swap_end_d = add_months(swap_start_inp, int(swap_term_y * 12))
        st.date_input("Swap End", value=swap_end_d, key="tix_swap_end", disabled=True)

    with p3:
        st.markdown(f"<div style='font-size:13px;color:#94a3b8;margin-bottom:4px'>Strike: <b style='color:#e2e8f0'>{strike_rate:.4f}%</b></div>", unsafe_allow_html=True)
        premium_bp = st.number_input("Premium (bp)", value=float(premium_bp) if premium_bp else 0.0,
            min_value=-9999.0, step=0.5, format="%.2f", key="tix_premium_bp")
        # Premium $ directly under bp
        premium_usd = (float(notional) if notional else 100.0) * 1_000_000 * premium_bp / 10_000
        st.metric("Premium $", f"{currency} {premium_usd:,.0f}")

    with p4:
        notional = st.number_input("Notional (MM)", value=float(notional) if notional else 100.0,
            min_value=1.0, step=25.0, format="%.0f", key="tix_notional")
        conv = "QQ (3M BBSW)" if swap_term_y <= 3 else "SS (6M BBSW)"
        st.caption(conv)
        # Swap rolls — first 4
        _rolls = [add_months(swap_start_inp, _freq_m * (i+1)) for i in range(4)]
        _roll_strs = [f"{r.day} {r.strftime('%b')}" for r in _rolls]
        st.markdown(f"<div style='font-size:12px;color:#94a3b8'>Rolls: {' · '.join(_roll_strs)}…</div>", unsafe_allow_html=True)

    pricing_ok = bool(expiry_date and swap_start and premium_bp)
    st.divider()

    # ── Trade details ─────────────────────────────────────────────────────────
    st.markdown("#### Trade Details")
    d1,d2,d3 = st.columns(3)
    with d1:
        option_type = st.selectbox("Option Type",
            ["Straddle","Payers","Receivers"],
            index=["Straddle","Payers","Receivers"].index(option_type_pre)
                  if option_type_pre in ["Straddle","Payers","Receivers"] else 0)
        settlement = st.selectbox("Settlement", ["Physical","Swap"], index=0,
            help="Physical → LCH Cleared Swap (AUD standard)")
    with d2:
        spot_or_fwd = st.selectbox("Premium Basis", ["Fwd","Spot"],
            help="Fwd = forward premium (AFMA s3.13)")
        clearhouse  = st.selectbox("Clearhouse", ["LCH","CME","Bilateral"])
    with d3:
        sef = st.checkbox("SEF", value=False)
        prem_auto = expiry_date_inp if settlement == "Physical" else expiry_date_inp + timedelta(days=1)
        premium_date = st.date_input("Premium Date", value=prem_auto,
            help="Auto: expiry (Physical) or T+1 (Cash). AFMA s3.11.3")

    st.divider()

    # ── Counterparties ────────────────────────────────────────────────────────
    st.markdown("#### Counterparties")
    bank_map   = {b["bank"]: b for b in all_banks}
    bank_names = sorted(bank_map.keys())

    def t_for(bank_id): return [t for t in all_traders   if t["bank_id"] == bank_id]
    def d_for(bank_id): return [d for d in all_divisions if d["bank_id"] == bank_id]

    col_b, col_s = st.columns(2)

    with col_b:
        st.markdown("**Buyer** *(premium payer)*")
        bb_name  = st.selectbox("Bank",   bank_names, key="bb")
        bb       = bank_map[bb_name]
        bt_map   = {f"{t['firstname']} {t['lastname']}": t for t in t_for(bb["bank_id"])}
        bt_name  = st.selectbox("Trader", sorted(bt_map) or ["—"], key="bt")
        bt       = bt_map.get(bt_name)
        # Division optional — only show if records exist
        bd_list  = d_for(bb["bank_id"])
        if bd_list:
            bd_map  = {d["name"]: d for d in bd_list}
            bd_name = st.selectbox("Division", sorted(bd_map), key="bd")
            bd      = bd_map.get(bd_name)
        else:
            st.caption("No division records — BIC used directly")
            bd = None
        b_brok = st.number_input("Brokerage (AUD)", 0.0, value=500.0, step=50.0, key="b_brok")

    with col_s:
        st.markdown("**Seller** *(premium receiver)*")
        sb_name  = st.selectbox("Bank",   bank_names, key="sb")
        sb_      = bank_map[sb_name]
        st_map   = {f"{t['firstname']} {t['lastname']}": t for t in t_for(sb_["bank_id"])}
        st_name  = st.selectbox("Trader", sorted(st_map) or ["—"], key="st_")
        st_      = st_map.get(st_name)
        # Division optional
        sd_list  = d_for(sb_["bank_id"])
        if sd_list:
            sd_map  = {d["name"]: d for d in sd_list}
            sd_name = st.selectbox("Division", sorted(sd_map), key="sd")
            sd      = sd_map.get(sd_name)
        else:
            st.caption("No division records — BIC used directly")
            sd = None
        s_brok = st.number_input("Brokerage (AUD)", 0.0, value=500.0, step=50.0, key="s_brok")

    # Division no longer required — just need bank + trader
    buyer_ok  = bool(bt)
    seller_ok = bool(st_)

    # BIC auto-resolve — fires as soon as both traders selected
    bic_b_code = bic_s_code = ""
    if buyer_ok and seller_ok:
        bic_b, bic_s = resolve_bic(
            bb["bank_id"], sb_["bank_id"],
            bt["trader_id"], st_["trader_id"],
            all_bic
        )
        bic_b_code = bic_b["markitbiccode"] if bic_b else "⚠️ Not found"
        bic_s_code = bic_s["markitbiccode"] if bic_s else "⚠️ Not found"
        st.divider()
        bc1, bc2 = st.columns(2)
        bc1.metric("BIC Buyer",  bic_b_code)
        bc2.metric("BIC Seller", bic_s_code)

    st.divider()

    # ── Generate ──────────────────────────────────────────────────────────────
    can_go = pricing_ok and buyer_ok and seller_ok

    if st.button("📄 Generate Ticket", type="primary",
                 disabled=not can_go, use_container_width=True):

        trade_id = f"SWN{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{swap_term}-{option_expiry}"

        ticket = {
            "trade_id":          trade_id,
            "timestamp":         datetime.utcnow().isoformat() + "Z",
            "currency":          currency,
            "option_type":       option_type,
            "option_expiry":     option_expiry,
            "option_expiry_y":   option_expiry_y,
            "swap_term":         swap_term,
            "swap_term_y":       swap_term_y,
            "expiry_date":       expiry_date,
            "swap_start_date":   swap_start,
            "premium_date":      premium_date,
            "strike_rate":       strike_rate,
            "premium_bp":        premium_bp,
            "notional":          notional,
            "settlement":        settlement,
            "spot_or_fwd":       spot_or_fwd,
            "clearhouse":        clearhouse,
            "sef":               sef,
            "buyer_bank_name":   bb_name,
            "buyer_ov_bank_id":  bb.get("short_code", bb_name),
            "buyer_trader_id":   bt["trader_id"],
            "buyer_trader_name": bt_name,
            "buyer_brokerage":   b_brok,
            "seller_bank_name":  sb_name,
            "seller_ov_bank_id": sb_.get("short_code", sb_name),
            "seller_trader_id":  st_["trader_id"],
            "seller_trader_name":st_name,
            "seller_brokerage":  s_brok,
            "bic_buyer":         bic_b_code,
            "bic_seller":        bic_s_code,
        }

        mw_json = build_mw_json(ticket)
        ok, err = save_to_supabase(ticket, mw_json)

        ss.last_ticket   = ticket
        ss.last_mw_json  = mw_json

        if ok:
            st.success(f"✅ Ticket saved — `{trade_id}`")
        else:
            st.warning(f"Supabase insert failed ({err}) — ticket shown below for manual use.")

    # ── Display ───────────────────────────────────────────────────────────────
    if hasattr(ss, "last_ticket"):
        ticket  = ss.last_ticket
        mw_json = ss.last_mw_json

        st.divider()
        st.markdown(f"#### `{ticket['trade_id']}`")

        rows = {
            "Option Type":      ticket["option_type"],
            "Expiry":           f"{ticket['option_expiry']} / {ticket['expiry_date']}",
            "Underlying":       f"{ticket['swap_term']} swap from {ticket['swap_start_date']}",
            "Strike":           f"{ticket['strike_rate']:.4f}%",
            "Premium":          f"{ticket['premium_bp']:.2f} bp ({ticket['spot_or_fwd']})",
            "Notional":         f"{ticket['currency']} {ticket['notional']}MM",
            "Settlement":       f"{ticket['settlement']} — {ticket['clearhouse']}",
            "Premium Date":     str(ticket["premium_date"]),
            "Buyer":            f"{ticket['buyer_trader_name']} [{ticket['buyer_bank_name']}]",
            "Seller":           f"{ticket['seller_trader_name']} [{ticket['seller_bank_name']}]",
            "Buyer Brokerage":  f"AUD {ticket['buyer_brokerage']:,.0f}",
            "Seller Brokerage": f"AUD {ticket['seller_brokerage']:,.0f}",
            "BIC Buyer":        ticket["bic_buyer"],
            "BIC Seller":       ticket["bic_seller"],
        }
        for k, v in rows.items():
            r1, r2 = st.columns([1,2])
            r1.markdown(f"**{k}**")
            r2.write(v)

        st.divider()
        st.markdown("#### MW JSON Payload")
        st.code(json.dumps(mw_json, indent=2), language="json")
        st.caption("Ready for FIXEdge/Citycom submission once MW credentials are loaded.")

    elif not can_go:
        missing = []
        if not pricing_ok:  missing.append("pricing (run pricer)")
        if not buyer_ok:    missing.append("buyer counterparty")
        if not seller_ok:   missing.append("seller counterparty")
        st.info("Complete: " + " · ".join(missing))
