// RateEdge - Main JavaScript

// API base URL
const API_BASE = window.location.origin;

// Load dashboard data on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStatistics();
    loadMarketRates();
});

// Load database statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/api/statistics`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.data;
            
            // Update stat cards
            document.getElementById('totalRecords').textContent = 
                (stats.total_records || 0).toLocaleString();
            document.getElementById('currencies').textContent = 
                stats.currencies || '-';
            document.getElementById('latestDate').textContent = 
                stats.latest_date || '-';
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// Load current market rates
async function loadMarketRates() {
    const currencies = [
        { id: 'audCard', currency: 'AUD', tenor: '10Y' },
        { id: 'usdCard', currency: 'USD', tenor: '10Y' },
        { id: 'jpyCard', currency: 'JPY', tenor: '10Y' },
        { id: 'nzdCard', currency: 'NZD', tenor: '10Y' }
    ];
    
    for (const curr of currencies) {
        try {
            const response = await fetch(
                `${API_BASE}/api/rates?currency=${curr.currency}&tenor=${curr.tenor}&limit=1`
            );
            const data = await response.json();
            
            if (data.success && data.data.length > 0) {
                const rate = (data.data[0].rate * 100).toFixed(2);
                const card = document.getElementById(curr.id);
                if (card) {
                    card.querySelector('.market-rate').textContent = `${rate}%`;
                }
            }
        } catch (error) {
            console.error(`Failed to load ${curr.currency} rate:`, error);
        }
    }
}

// Utility function to format dates
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-AU', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

// Utility function to format rates
function formatRate(rate) {
    return (rate * 100).toFixed(4) + '%';
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 2rem;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
