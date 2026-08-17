from fpdf import FPDF
import datetime

class MaintenanceReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Industrial Maintenance Diagnostic Report', border=False, ln=True, align='C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', border=False, ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(prod_type, air_temp, proc_temp, rot_speed, torque, tool_wear, failure_prob, reasons):
    pdf = MaintenanceReport()
    pdf.add_page()
    pdf.set_font('Helvetica', '', 12)
    
    # Risk Summary Header
    if failure_prob >= 50.0:
        pdf.set_text_color(200, 0, 0)
        status_text = "CRITICAL RISK - FAILURE DETECTED"
    else:
        pdf.set_text_color(0, 128, 0)
        status_text = "NORMAL OPERATING CONDITIONS"
        
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, f"Status: {status_text}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f"Failure Probability: {failure_prob:.2f}%", ln=True)
    pdf.ln(5)
    
    # Telemetry Table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "Equipment Telemetry Summary:", ln=True)
    pdf.set_font('Helvetica', '', 10)
    
    data = [
        ("Product Quality Type", str(prod_type)),
        ("Air Temperature [K]", f"{air_temp} K"),
        ("Process Temperature [K]", f"{proc_temp} K"),
        ("Rotational Speed [rpm]", f"{rot_speed} RPM"),
        ("Torque [Nm]", f"{torque} Nm"),
        ("Tool Wear [min]", f"{tool_wear} min"),
    ]
    
    for label, val in data:
        pdf.cell(90, 7, label, border=1)
        pdf.cell(90, 7, val, border=1, ln=True)
        
    pdf.ln(8)
    
    # Failure Mode Diagnostics
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "Diagnostic Notes & Root Causes:", ln=True)
    pdf.set_font('Helvetica', '', 10)
    
    if reasons:
        for r in reasons:
            clean_r = r.replace('⚠️', '[WARNING]').replace('**', '')
            pdf.multi_cell(0, 7, f"- {clean_r}")
    else:
        pdf.cell(0, 7, "No critical physical thresholds exceeded.", ln=True)
        
    return bytes(pdf.output())