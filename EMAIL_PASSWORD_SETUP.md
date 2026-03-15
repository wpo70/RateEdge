# Email Password Setup - Azure Environment Variable

## Quick Setup Instructions

### Azure Portal (Web Interface)

1. **Navigate to your App Service**
   - Go to: https://portal.azure.com
   - Find: `rateedge-options` (or your app service name)

2. **Open Configuration**
   - Left menu → **Configuration**
   - Click **Application settings** tab

3. **Add Environment Variable**
   - Click **+ New application setting**
   - **Name**: `EMAIL_PASSWORD`
   - **Value**: Your wpo@rateedge.au password
   - Click **OK**

4. **Save Changes**
   - Click **Save** at the top
   - Click **Continue** to confirm restart

### Azure CLI (Command Line)

```bash
az webapp config appsettings set \
  --name rateedge-options \
  --resource-group rateedge-rg \
  --settings EMAIL_PASSWORD="your_password_here"
```

## How It Works

### In the Code:
```python
default_password = os.getenv("EMAIL_PASSWORD", "")
smtp_password = st.text_input("SMTP Password", value=default_password, type="password")
```

### Behavior:
- ✅ **If EMAIL_PASSWORD is set**: Password auto-fills, ready to send
- ⚠️ **If EMAIL_PASSWORD is not set**: Must enter password manually each time

### UI Indicator:
- **Green checkmark**: ✅ Password loaded from environment variable
- **Warning**: ⚠️ Set EMAIL_PASSWORD environment variable in Azure to auto-fill

## Security Notes

1. **Password is never exposed** - stored securely in Azure
2. **Only accessible** by your App Service
3. **Can still override** in UI if needed (e.g., testing different accounts)
4. **Password is masked** in UI (type="password")

## Testing

After setting the variable:
1. Restart the app service (or it restarts automatically after save)
2. Go to Vol Export tab
3. Expand "⚙️ Email Settings (SMTP)"
4. You should see: ✅ Password loaded from environment variable EMAIL_PASSWORD
5. Password field will show dots (••••••••) but is populated

## Troubleshooting

**Password not loading?**
- Check variable name is exactly: `EMAIL_PASSWORD` (case-sensitive)
- Verify app service was restarted after adding
- Check in Azure Portal → Configuration that variable exists

**Still need to enter password?**
- Try refreshing the Streamlit app (F5)
- Check browser console for errors
- Verify environment variable in app service logs

## Office 365 Settings

Current defaults:
- **SMTP Server**: smtp.office365.com
- **SMTP Port**: 587
- **SMTP Username**: wpo@rateedge.au
- **Use TLS**: Enabled

## App-Specific Password (If MFA Enabled)

If your Office 365 account has multi-factor authentication:

1. Go to: https://account.microsoft.com/security
2. Navigate to: **Security** → **App passwords**
3. Click: **Create a new app password**
4. Name it: "RateEdge Options Platform"
5. Copy the generated password
6. Use this password in `EMAIL_PASSWORD` variable

Benefits:
- More secure than main password
- Can revoke without changing main password
- Bypasses MFA for automated sending
