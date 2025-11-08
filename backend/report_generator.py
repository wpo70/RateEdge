"""
PDF Report Generator for Swap Rates
Creates professional PDF reports with charts and statistics
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import matplotlib.pyplot as plt
import io
from pathlib import Path


class SwapRateReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self, db_manager, analytics):
        self.db_manager = db_manager
        self.analytics = analytics
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate_market_report(self, currency, output_path, tenors=None):
        """
        Generate comprehensive market report
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph(f"{currency} Swap Rate Market Report", self.styles['CustomTitle'])
        story.append(title)
        
        subtitle = Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.styles['Normal']
        )
        story.append(subtitle)
        story.append(Spacer(1, 0.3*inch))
        
        # Get tenors
        if not tenors:
            tenors = self.db_manager.get_available_tenors(currency)
        
        # Latest Rates Summary
        story.append(Paragraph("Current Market Rates", self.styles['SectionHeader']))
        latest_table = self._create_latest_rates_table(currency, tenors)
        if latest_table:
            story.append(latest_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Yield Curve Chart
        story.append(Paragraph("Yield Curve", self.styles['SectionHeader']))
        curve_chart = self._create_yield_curve_chart(currency)
        if curve_chart:
            story.append(curve_chart)
            story.append(Spacer(1, 0.2*inch))
        
        # Statistics for key tenors
        key_tenors = self._select_key_tenors(tenors)
        for tenor in key_tenors[:3]:  # Limit to 3 tenors for report length
            story.append(PageBreak())
            story.append(Paragraph(f"{tenor} Detailed Analysis", self.styles['SectionHeader']))
            
            stats = self.analytics.get_rate_statistics(currency, tenor)
            if stats:
                stats_table = self._create_statistics_table(stats, tenor)
                story.append(stats_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Historical chart for this tenor
            history_chart = self._create_historical_chart(currency, tenor, days=365)
            if history_chart:
                story.append(history_chart)
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _create_latest_rates_table(self, currency, tenors):
        """Create table of latest rates"""
        latest_rates = self.db_manager.get_latest_rates(currency)
        
        if not latest_rates:
            return None
        
        # Filter to requested tenors
        if tenors:
            latest_rates = [r for r in latest_rates if r.tenor in tenors]
        
        data = [['Tenor', 'Rate (%)', 'Date']]
        
        for rate in latest_rates:
            data.append([
                rate.tenor,
                f"{rate.rate * 100:.4f}",
                rate.date.strftime('%Y-%m-%d')
            ])
        
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_statistics_table(self, stats, tenor):
        """Create statistics summary table"""
        data = [
            ['Metric', 'Value'],
            ['Current Rate', f"{stats['current']:.4f}%"],
            ['Mean', f"{stats['mean']:.4f}%"],
            ['Median', f"{stats['median']:.4f}%"],
            ['Std Dev', f"{stats['std_dev']:.4f}%"],
            ['Min', f"{stats['min']:.4f}%"],
            ['Max', f"{stats['max']:.4f}%"],
            ['Range', f"{stats['range']:.4f}%"],
        ]
        
        if 'change_1d' in stats:
            data.extend([
                ['Change (1 Day)', f"{stats['change_1d']:+.4f}%"],
                ['Change (1 Week)', f"{stats['change_1w']:+.4f}%"],
                ['Change (1 Month)', f"{stats['change_1m']:+.4f}%"],
            ])
        
        table = Table(data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_yield_curve_chart(self, currency):
        """Create yield curve chart as image"""
        latest_rates = self.db_manager.get_latest_rates(currency)
        
        if not latest_rates:
            return None
        
        from database_models import tenor_sort_key
        
        # Sort and prepare data
        sorted_rates = sorted(latest_rates, key=lambda x: tenor_sort_key(x.tenor))
        tenors = [r.tenor for r in sorted_rates]
        rates = [r.rate * 100 for r in sorted_rates]
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tenors, rates, marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.set_xlabel('Tenor', fontsize=11)
        ax.set_ylabel('Rate (%)', fontsize=11)
        ax.set_title(f'{currency} Yield Curve', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Create ReportLab image
        img = Image(img_buffer, width=5*inch, height=3.3*inch)
        return img
    
    def _create_historical_chart(self, currency, tenor, days=365):
        """Create historical rate chart"""
        from datetime import timedelta
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        rates = self.db_manager.get_rates(currency, tenor, start_date, end_date)
        
        if not rates:
            return None
        
        dates = [r.date for r in reversed(rates)]
        values = [r.rate * 100 for r in reversed(rates)]
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(dates, values, linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Rate (%)', fontsize=10)
        ax.set_title(f'{currency} {tenor} - Last {days} Days', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        img = Image(img_buffer, width=5*inch, height=2.9*inch)
        return img
    
    def _select_key_tenors(self, tenors):
        """Select key tenors for detailed analysis"""
        # Prioritize common benchmark tenors
        priority = ['2Y', '5Y', '10Y', '30Y', '1Y', '3Y', '7Y']
        
        selected = []
        for t in priority:
            if t in tenors:
                selected.append(t)
        
        # Add any remaining tenors
        for t in tenors:
            if t not in selected:
                selected.append(t)
        
        return selected
