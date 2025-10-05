"""
Automated tax risk analysis and reporting
"""
from typing import Dict, List
from utils.logger import setup_logger
from utils.rag_engine import RAGEngine

logger = setup_logger("tax_analyzer")

class TaxAnalyzer:
    """Generate comprehensive tax risk analysis reports"""
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        
    def analyze_document(self) -> Dict:
        """Run comprehensive tax analysis across all sections"""
        
        logger.info("Starting comprehensive tax analysis")
        
        # Define analysis questions for each section
        analysis_sections = {
            "audit_outcomes": [
                "What are the tax audit outcomes and indicators mentioned?",
                "Are there any hard labels or historical audit findings?",
                "What is the audit risk score?"
            ],
            "business_analysis": [
                "What is the effective tax rate mentioned?",
                "What are the tax-adjusted returns or IRR mentioned?",
                "What scenarios are analyzed (base case, P75, P90, P95)?",
                "What are the key tax metrics and financial figures?"
            ],
            "escalation_flags": [
                "What are the critical tax escalation flags or risk drivers?",
                "Are there any material tax issues requiring partner attention?"
            ],
            "executive_summary": [
                "Provide a comprehensive executive summary of the key tax takeaways",
                "What are the main tax risks and opportunities?",
                "What are the quantified risk insights?"
            ],
            "balance_sheet": [
                "What balance sheet and tax metrics are mentioned?",
                "What tax liabilities or assets are disclosed?"
            ],
            "transaction_structure": [
                "What is the transaction structure (merger, acquisition, deal type)?",
                "What jurisdictions are involved?"
            ],
            "investment_analysis": [
                "What is the share class IRR or tax-adjusted IRR?",
                "What investment recommendations are provided?",
                "What are the reserve recommendations?"
            ],
            "tax_contingencies": [
                "What tax contingencies or reserves are mentioned?",
                "What is the expected tax contingency distribution (P50, P75, P90, P95)?",
                "What methodologies are used (Log-Normal, Conservative, Triangular)?"
            ]
        }
        
        report = {}
        
        for section, questions in analysis_sections.items():
            logger.info(f"Analyzing section: {section}")
            section_results = []
            
            for question in questions:
                response = self.rag.query(question)
                section_results.append({
                    "question": question,
                    "answer": response["answer"],
                    "sources": response["sources"]
                })
            
            report[section] = section_results
        
        logger.info("Tax analysis complete")
        return report
    
    def generate_summary_report(self, analysis: Dict) -> str:
        """Generate formatted summary report"""
        
        report = "# M&A TAX RISK ASSESSMENT REPORT\n\n"
        
        section_titles = {
            "audit_outcomes": "🔍 Tax Audit Outcomes & Indicators",
            "business_analysis": "📊 Business Analysis for VC/PE",
            "escalation_flags": "⚠️ Escalation Flags & Risk Drivers",
            "executive_summary": "📋 Executive Summary & Key Takeaways",
            "balance_sheet": "💰 Balance Sheet & Tax Metrics",
            "transaction_structure": "🤝 Transaction Structure",
            "investment_analysis": "📈 Integrated Investment Analysis",
            "tax_contingencies": "💼 Tax Contingency Distribution"
        }
        
        for section, title in section_titles.items():
            report += f"\n## {title}\n\n"
            
            if section in analysis:
                for item in analysis[section]:
                    report += f"**Q: {item['question']}**\n\n"
                    report += f"{item['answer']}\n\n"
                    
                    if item['sources']:
                        report += "_Sources:_\n"
                        for idx, source in enumerate(item['sources'][:2], 1):
                            report += f"- Source {idx}: {source['content'][:150]}...\n"
                        report += "\n"
            
            report += "---\n"
        
        return report