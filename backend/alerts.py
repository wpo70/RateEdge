"""
Alerts and Notifications System for Swap Rates
Monitor rates and trigger alerts based on conditions
"""
import json
import os
from datetime import datetime
from pathlib import Path


class AlertManager:
    """Manage rate alerts and notifications"""
    
    def __init__(self, db_manager, alerts_file='alerts.json'):
        self.db_manager = db_manager
        self.alerts_dir = Path(__file__).parent.parent / 'database'
        self.alerts_dir.mkdir(exist_ok=True)
        self.alerts_file = self.alerts_dir / alerts_file
        self.alerts = self.load_alerts()
    
    def load_alerts(self):
        """Load alerts from file"""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_alerts(self):
        """Save alerts to file"""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def add_alert(self, alert_config):
        """
        Add a new alert
        
        alert_config should contain:
        - name: Alert name
        - currency: AUD or NZD
        - tenor: e.g., 5Y
        - condition: 'above', 'below', 'crosses_above', 'crosses_below', 'change'
        - threshold: threshold value
        - enabled: True/False
        """
        alert = {
            'id': len(self.alerts) + 1,
            'created': datetime.now().isoformat(),
            'last_checked': None,
            'last_triggered': None,
            'trigger_count': 0,
            **alert_config
        }
        
        self.alerts.append(alert)
        self.save_alerts()
        return alert
    
    def remove_alert(self, alert_id):
        """Remove an alert by ID"""
        self.alerts = [a for a in self.alerts if a['id'] != alert_id]
        self.save_alerts()
    
    def update_alert(self, alert_id, updates):
        """Update an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert.update(updates)
                self.save_alerts()
                return alert
        return None
    
    def enable_alert(self, alert_id):
        """Enable an alert"""
        return self.update_alert(alert_id, {'enabled': True})
    
    def disable_alert(self, alert_id):
        """Disable an alert"""
        return self.update_alert(alert_id, {'enabled': False})
    
    def check_alerts(self):
        """
        Check all enabled alerts and return triggered ones
        Returns list of triggered alerts with details
        """
        triggered = []
        
        for alert in self.alerts:
            if not alert.get('enabled', False):
                continue
            
            # Get latest rate
            rates = self.db_manager.get_latest_rates(alert['currency'])
            current_rate = None
            
            for rate in rates:
                if rate.tenor == alert['tenor']:
                    current_rate = rate.rate * 100  # Convert to percentage
                    break
            
            if current_rate is None:
                continue
            
            # Check condition
            is_triggered = False
            message = ""
            
            condition = alert['condition']
            threshold = alert['threshold']
            
            if condition == 'above' and current_rate > threshold:
                is_triggered = True
                message = f"{alert['tenor']} rate ({current_rate:.2f}%) is above {threshold}%"
            
            elif condition == 'below' and current_rate < threshold:
                is_triggered = True
                message = f"{alert['tenor']} rate ({current_rate:.2f}%) is below {threshold}%"
            
            elif condition in ['crosses_above', 'crosses_below']:
                # Need previous rate to check crossing
                all_rates = self.db_manager.get_rates(
                    currency=alert['currency'],
                    tenor=alert['tenor']
                )
                
                if len(all_rates) >= 2:
                    prev_rate = all_rates[1].rate * 100
                    
                    if condition == 'crosses_above':
                        if prev_rate <= threshold < current_rate:
                            is_triggered = True
                            message = f"{alert['tenor']} crossed above {threshold}% (was {prev_rate:.2f}%, now {current_rate:.2f}%)"
                    
                    elif condition == 'crosses_below':
                        if prev_rate >= threshold > current_rate:
                            is_triggered = True
                            message = f"{alert['tenor']} crossed below {threshold}% (was {prev_rate:.2f}%, now {current_rate:.2f}%)"
            
            elif condition == 'change':
                # Alert on absolute change
                all_rates = self.db_manager.get_rates(
                    currency=alert['currency'],
                    tenor=alert['tenor']
                )
                
                if len(all_rates) >= 2:
                    prev_rate = all_rates[1].rate * 100
                    change = abs(current_rate - prev_rate)
                    
                    if change >= threshold:
                        is_triggered = True
                        direction = "increased" if current_rate > prev_rate else "decreased"
                        message = f"{alert['tenor']} {direction} by {change:.2f}% (now {current_rate:.2f}%)"
            
            # Update alert status
            alert['last_checked'] = datetime.now().isoformat()
            
            if is_triggered:
                alert['last_triggered'] = datetime.now().isoformat()
                alert['trigger_count'] = alert.get('trigger_count', 0) + 1
                
                triggered.append({
                    'alert': alert,
                    'current_rate': current_rate,
                    'message': message
                })
        
        self.save_alerts()
        return triggered
    
    def get_alerts(self, enabled_only=False):
        """Get all alerts or only enabled ones"""
        if enabled_only:
            return [a for a in self.alerts if a.get('enabled', False)]
        return self.alerts
    
    def get_alert_history(self, alert_id, limit=10):
        """Get history of when an alert was triggered (placeholder for future)"""
        # In a full implementation, this would query a separate history table
        alert = next((a for a in self.alerts if a['id'] == alert_id), None)
        if alert:
            return {
                'alert_id': alert_id,
                'trigger_count': alert.get('trigger_count', 0),
                'last_triggered': alert.get('last_triggered'),
                'last_checked': alert.get('last_checked')
            }
        return None


class AlertConditions:
    """Constants for alert conditions"""
    ABOVE = 'above'
    BELOW = 'below'
    CROSSES_ABOVE = 'crosses_above'
    CROSSES_BELOW = 'crosses_below'
    CHANGE = 'change'
    
    @classmethod
    def get_all(cls):
        return [cls.ABOVE, cls.BELOW, cls.CROSSES_ABOVE, cls.CROSSES_BELOW, cls.CHANGE]
    
    @classmethod
    def get_description(cls, condition):
        descriptions = {
            cls.ABOVE: "Rate is above threshold",
            cls.BELOW: "Rate is below threshold",
            cls.CROSSES_ABOVE: "Rate crosses above threshold",
            cls.CROSSES_BELOW: "Rate crosses below threshold",
            cls.CHANGE: "Rate changes by threshold amount"
        }
        return descriptions.get(condition, "Unknown condition")
