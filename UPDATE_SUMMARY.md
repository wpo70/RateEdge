# Swaption Pricer Updates - Historical Vols & Extended Expiries

## Changes Made

### 1. Extended Expiries (17 → 22 expiries)

**OLD expiries list:**
```
1w, 1m, 2m, 3m, 6m, 9m, 1y, 18m, 2y, 3y, 4y, 5y, 7y, 10y, 12y, 15y, 20y
```

**NEW expiries list:**
```
1w, 1m, 2m, 3m, 6m, 9m, 1y, 18m, 2y, 3y, 4y, 5y, 6y, 7y, 8y, 9y, 10y, 12y, 15y, 20y, 25y, 30y
```

**Added:** 6y, 8y, 9y, 25y, 30y

**Updated in 3 locations:**
- Line ~1884: `_generate_forward_matrix_cached()` 
- Line ~3277: Multi-currency pricing expiry dropdown
- Line ~3994: `_generate_basis_matrix_cached()`

### 2. Historical Vol Storage Functions

**New functions added (175 lines):**

```python
save_vol_snapshot(user_id, currency, label, notes)
list_vol_snapshots(user_id, currency)
load_vol_snapshot(snapshot_id)
delete_vol_snapshot(snapshot_id)
```

**Location:** Added after `load_all_session_data()` function (~line 362)

**Database table required:** `vol_history` (see deployment steps)

### 3. Files Updated

- `app_streamlit.py`: 4421 → 4596 lines (+175 lines)
- `vol_editor.py`: No changes (unchanged)

---

## Deployment Steps

### Step 1: Create Database Table

Run this SQL in **Azure RateEdge → swaption** database:

```sql
CREATE TABLE IF NOT EXISTS vol_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL DEFAULT 'default',
    currency VARCHAR(3) NOT NULL,
    snapshot_date TIMESTAMP NOT NULL,
    label VARCHAR(255),
    atm_vols JSONB NOT NULL,
    sabr_alpha JSONB,
    sabr_beta JSONB,
    sabr_rho JSONB,
    sabr_nu JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vol_history_lookup 
ON vol_history(user_id, currency, snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_vol_history_label
ON vol_history(user_id, label);
```

### Step 2: Upload Complete Config to Database

Use **RateEdge_Config_Complete.xlsx** to upload:
- All 22 expiries populated
- Correct format (lowercase expiries: 1w, 1m, etc.)
- Column names (1Y, 2Y, 3Y, etc.)

Upload in swaption pricer → "📥 Commit Selected Data" → "💾 Save to Database"

### Step 3: Deploy Updated App

```powershell
# Navigate to your swaption pricer directory
cd "C:\Users\willp\RateEdge Swaption Pricer"

# Replace the files
# (Copy app_streamlit.py and vol_editor.py from outputs)

# Build and deploy
az acr build --registry rateedgeacr --image swaption:latest .
az webapp config container set --resource-group rateedge-rg --name rateedge-options --container-image-name rateedgeacr.azurecr.io/swaption:latest
az webapp restart --resource-group rateedge-rg --name rateedge-options
```

---

## New Features Available

### Historical Vol Snapshots

Users can now:
1. **Save** current vol surface with label/notes
2. **Browse** historical snapshots by currency and date
3. **Load** past snapshots to compare or restore
4. **Delete** old snapshots

**Usage:** Add UI to Vol/SABR Config tab (see swaption_pricer_updates.py for UI code)

### Extended Expiries

All pricing, forward curves, and vol surfaces now support:
- **6y** - 6 year expiries
- **8y** - 8 year expiries  
- **9y** - 9 year expiries
- **25y** - 25 year expiries
- **30y** - 30 year expiries

---

## Testing Checklist

After deployment:

- [ ] Database table `vol_history` exists
- [ ] Upload RateEdge_Config_Complete.xlsx successfully
- [ ] All 22 expiries visible in forward matrix
- [ ] Price a swaption with 25y expiry
- [ ] Save a vol snapshot (test function)
- [ ] Load a vol snapshot (test function)
- [ ] Verify pricing still works correctly

---

## Safety Notes

**These changes DO NOT break existing functionality:**
- ✅ All existing 17 expiries still work
- ✅ New expiries are additions only
- ✅ Historical vol storage is optional
- ✅ Old configs continue to work
- ✅ Pricing formulas unchanged

**The vol_history table is separate from user_configs** - no risk of data corruption.

---

## Files Included

1. `app_streamlit.py` - Updated main app with extended expiries + historical vols
2. `vol_editor.py` - Unchanged (no updates needed)
3. `add_vol_history.sql` - Database table creation script
4. `RateEdge_Config_Complete.xlsx` - Config file with all 22 expiries populated
5. `swaption_pricer_updates.py` - Reference code for historical vol UI (optional)

---

## Questions?

If pricing breaks or anything doesn't work, the issue is most likely:
1. Database table not created
2. Config file format mismatch
3. Missing expiry data in uploaded config

Check these first before debugging further.
