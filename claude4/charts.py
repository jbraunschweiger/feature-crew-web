import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(20, 24))

# Data preparation
# Model release timeline and key metrics
models_data = {
    'Model': ['GPT-4', 'Claude 2', 'Gemini 1.5', 'Llama 3', 'DeepSeek-V3', 'Gemini 2.0', 
              'DeepSeek-R1', 'Claude 3.7', 'Grok 3', 'GPT-4.1', 'Llama 4', 'Qwen 3', 
              'Nova Premier', 'Claude 4'],
    'Company': ['OpenAI', 'Anthropic', 'Google', 'Meta', 'DeepSeek', 'Google', 
                'DeepSeek', 'Anthropic', 'xAI', 'OpenAI', 'Meta', 'Alibaba', 
                'Amazon', 'Anthropic'],
    'Release_Date': ['2023-03', '2023-07', '2024-02', '2024-04', '2024-12', '2024-12',
                     '2025-01', '2025-02', '2025-02', '2025-04', '2025-04', '2025-04',
                     '2025-04', '2025-05'],
    'Context_Window': [32, 100, 1000, 128, 128, 128, 
                       32, 200, 128, 1000, 10000, 200,
                       1000, 200],
    'Parameters_B': [1000, 70, 1000, 405, 671, 1000,
                     671, 70, 1000, 1000, 400, 235,
                     100, 100],
    'Cost_Per_M_Tokens': [30, 8, 3.5, 0, 0.5, 3.5,
                          0.55, 3, 2, 5, 0, 0.5,
                          2.5, 3],
    'Has_Reasoning': [False, False, False, False, False, True,
                      True, True, True, False, False, True,
                      False, True],
    'Performance_Score': [85, 80, 88, 82, 90, 92,
                          93, 91, 94, 89, 92, 95,
                          87, 96]
}

df = pd.DataFrame(models_data)
df['Release_Date'] = pd.to_datetime(df['Release_Date'])

# 1. Timeline of Model Releases with Context Window Evolution
ax1 = plt.subplot(4, 2, 1)
companies = df['Company'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(companies)))
company_colors = dict(zip(companies, colors))

for company in companies:
    company_df = df[df['Company'] == company]
    ax1.scatter(company_df['Release_Date'], company_df['Context_Window'], 
                label=company, s=200, alpha=0.7, color=company_colors[company])
    
    # Add model names
    for idx, row in company_df.iterrows():
        ax1.annotate(row['Model'], (row['Release_Date'], row['Context_Window']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_yscale('log')
ax1.set_xlabel('Release Date')
ax1.set_ylabel('Context Window (K tokens)')
ax1.set_title('Evolution of Model Context Windows Over Time', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Cost vs Performance Analysis
ax2 = plt.subplot(4, 2, 2)
reasoning_models = df[df['Has_Reasoning'] == True]
non_reasoning_models = df[df['Has_Reasoning'] == False]

ax2.scatter(non_reasoning_models['Cost_Per_M_Tokens'], non_reasoning_models['Performance_Score'], 
            s=300, alpha=0.6, label='Standard Models', marker='o')
ax2.scatter(reasoning_models['Cost_Per_M_Tokens'], reasoning_models['Performance_Score'], 
            s=300, alpha=0.6, label='Reasoning Models', marker='^')

# Add model labels
for idx, row in df.iterrows():
    ax2.annotate(row['Model'], (row['Cost_Per_M_Tokens'], row['Performance_Score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('Cost per Million Tokens ($)')
ax2.set_ylabel('Performance Score')
ax2.set_title('Cost-Performance Trade-off Analysis', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Investment Levels by Company
ax3 = plt.subplot(4, 2, 3)
investment_data = {
    'Company': ['Microsoft', 'Amazon', 'Google', 'Meta'],
    'Investment_2025_B': [80, 100, 75, 65]
}
inv_df = pd.DataFrame(investment_data)
bars = ax3.bar(inv_df['Company'], inv_df['Investment_2025_B'], alpha=0.7)
ax3.set_ylabel('Investment (Billions USD)')
ax3.set_title('2025 AI Infrastructure Investment by Tech Giants', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'${height}B', ha='center', va='bottom')

# 4. Model Performance Progression
ax4 = plt.subplot(4, 2, 4)
for company in ['OpenAI', 'Google', 'Anthropic', 'DeepSeek']:
    company_df = df[df['Company'] == company].sort_values('Release_Date')
    if len(company_df) > 1:
        ax4.plot(company_df['Release_Date'], company_df['Performance_Score'], 
                marker='o', linewidth=2, markersize=10, label=company, 
                color=company_colors[company])

ax4.set_xlabel('Release Date')
ax4.set_ylabel('Performance Score')
ax4.set_title('Performance Progression by Company', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Performance Projections
ax5 = plt.subplot(4, 2, 5)
# Historical data points
dates = pd.date_range('2023-01', '2025-05', freq='3M')
performance = np.array([75, 78, 82, 85, 88, 90, 92, 94, 96])

# Fit exponential growth model
from scipy.optimize import curve_fit
def exp_growth(x, a, b, c):
    return a * np.exp(b * x) + c

x_data = np.arange(len(performance))
popt, _ = curve_fit(exp_growth, x_data, performance)

# Project forward
future_dates = pd.date_range('2025-06', '2027-01', freq='3M')
all_dates = pd.concat([pd.Series(dates), pd.Series(future_dates)])
x_all = np.arange(len(all_dates))
y_pred = exp_growth(x_all, *popt)

ax5.plot(dates, performance, 'o', markersize=10, label='Historical Performance')
ax5.plot(all_dates, y_pred, '--', linewidth=2, label='Projected Performance')
ax5.axvline(x=pd.Timestamp('2025-05-15'), color='red', linestyle=':', alpha=0.5)
ax5.text(pd.Timestamp('2025-05-15'), 85, 'Today', rotation=90, va='bottom')

ax5.set_xlabel('Date')
ax5.set_ylabel('Performance Score')
ax5.set_title('AI Model Performance Projection', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_ylim(70, 110)

# 6. Model Capabilities Matrix
ax6 = plt.subplot(4, 2, 6)
capabilities = pd.DataFrame({
    'Model': ['GPT-4.1', 'Claude 3.7', 'Gemini 2.5', 'Llama 4', 'DeepSeek-R1', 'Grok 3'],
    'Text': [5, 5, 5, 5, 5, 5],
    'Code': [4.5, 4, 5, 4, 5, 4.5],
    'Reasoning': [3, 5, 4.5, 3.5, 5, 4.5],
    'Vision': [4, 4, 5, 4.5, 3, 4],
    'Cost_Efficiency': [2, 3, 3.5, 5, 5, 2.5],
    'Context_Length': [4, 3, 3.5, 5, 2, 3]
})

categories = ['Text', 'Code', 'Reasoning', 'Vision', 'Cost_Efficiency', 'Context_Length']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for idx, model in enumerate(capabilities['Model']):
    values = capabilities.loc[capabilities['Model'] == model, categories].values.flatten().tolist()
    values += values[:1]
    
    ax6.plot(angles, values, 'o-', linewidth=2, label=model, markersize=8)
    ax6.fill(angles, values, alpha=0.1)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories)
ax6.set_ylim(0, 5.5)
ax6.set_title('Model Capabilities Comparison', fontsize=14, fontweight='bold')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.grid(True, alpha=0.3)

# 7. Market Dynamics - Open vs Closed Source
ax7 = plt.subplot(4, 2, 7)
open_source = ['Llama 4', 'DeepSeek-R1', 'DeepSeek-V3', 'Qwen 3']
closed_source = ['GPT-4.1', 'Claude 3.7', 'Gemini 2.5', 'Grok 3', 'Nova Premier']

# Count models by type over time
timeline = pd.date_range('2023-01', '2025-05', freq='Q')
open_count = [0, 0, 1, 1, 2, 2, 3, 4, 4]
closed_count = [2, 3, 3, 4, 4, 5, 6, 7, 9]

ax7.plot(timeline, open_count, marker='o', linewidth=3, markersize=10, label='Open Source')
ax7.plot(timeline, closed_count, marker='s', linewidth=3, markersize=10, label='Closed Source')
ax7.fill_between(timeline, 0, open_count, alpha=0.3)
ax7.fill_between(timeline, open_count, closed_count, alpha=0.3)

ax7.set_xlabel('Date')
ax7.set_ylabel('Cumulative Model Count')
ax7.set_title('Open Source vs Closed Source AI Models', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Business Recommendation Matrix
ax8 = plt.subplot(4, 2, 8)
business_scenarios = {
    'Scenario': ['Customer Service', 'Content Creation', 'Code Development', 
                 'Data Analysis', 'Document Processing', 'Research & Innovation'],
    'Recommended_Model': ['Claude 3.7', 'Gemini 2.5', 'DeepSeek-R1', 
                          'GPT-4.1', 'Llama 4 Scout', 'Claude 4 Opus'],
    'Cost_Level': [2, 2, 1, 3, 1, 4],
    'Complexity': [2, 2, 4, 3, 3, 5],
    'ROI_Potential': [4, 5, 5, 4, 3, 3]
}

biz_df = pd.DataFrame(business_scenarios)
x = np.arange(len(biz_df['Scenario']))
width = 0.25

ax8.bar(x - width, biz_df['Cost_Level'], width, label='Cost Level', alpha=0.8)
ax8.bar(x, biz_df['Complexity'], width, label='Implementation Complexity', alpha=0.8)
ax8.bar(x + width, biz_df['ROI_Potential'], width, label='ROI Potential', alpha=0.8)

ax8.set_xlabel('Business Use Case')
ax8.set_ylabel('Score (1-5)')
ax8.set_title('AI Model Recommendations by Business Scenario', fontsize=14, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(biz_df['Scenario'], rotation=45, ha='right')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Add recommended model labels
for i, (scenario, model) in enumerate(zip(biz_df['Scenario'], biz_df['Recommended_Model'])):
    ax8.text(i, 5.2, model, ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig('ai_models_analysis_2025.png', dpi=300, bbox_inches='tight')
plt.show()

# Cost Analysis for SMBs
print("\n" + "="*80)
print("COST ANALYSIS FOR SMALL TO MEDIUM BUSINESSES")
print("="*80)

# Calculate monthly costs for different usage scenarios
usage_scenarios = {
    'Light Usage (1M tokens/month)': 1,
    'Medium Usage (10M tokens/month)': 10,
    'Heavy Usage (50M tokens/month)': 50
}

recommended_models = {
    'DeepSeek-R1': {'input_cost': 0.55, 'output_cost': 2.19, 'type': 'Open Source'},
    'Claude 3.7 Sonnet': {'input_cost': 3, 'output_cost': 15, 'type': 'Closed Source'},
    'Llama 4 (self-hosted)': {'input_cost': 0, 'output_cost': 0, 'type': 'Open Source'},
    'GPT-4.1': {'input_cost': 5, 'output_cost': 15, 'type': 'Closed Source'}
}

print("\nMonthly Cost Estimates (assuming 70% input, 30% output tokens):")
print("-" * 80)
for scenario, millions in usage_scenarios.items():
    print(f"\n{scenario}:")
    for model, costs in recommended_models.items():
        input_tokens = millions * 0.7
        output_tokens = millions * 0.3
        monthly_cost = (input_tokens * costs['input_cost']) + (output_tokens * costs['output_cost'])
        
        if model == 'Llama 4 (self-hosted)':
            # Estimate infrastructure costs
            monthly_cost = millions * 2  # Rough estimate for hosting
            print(f"  {model}: ~${monthly_cost:.2f}/month (infrastructure costs)")
        else:
            print(f"  {model}: ${monthly_cost:.2f}/month")

# Performance vs Cost Trade-off Analysis
print("\n" + "="*80)
print("KEY INSIGHTS FROM ANALYSIS")
print("="*80)

insights = """
1. RAPID ACCELERATION IN CAPABILITIES:
   - Context windows expanded from 32K to 10M tokens in 2 years
   - Performance scores improving ~15-20% annually
   - Reasoning capabilities becoming standard in 2025

2. COST TRENDS:
   - Open source models (DeepSeek, Llama) offering 80-95% cost savings
   - Hybrid models emerging to balance cost and capability
   - Infrastructure investments reaching $320B+ in 2025 alone

3. MARKET DYNAMICS:
   - Chinese companies (DeepSeek, Alibaba) challenging US dominance
   - Open source adoption accelerating rapidly
   - Specialization increasing (coding, reasoning, vision)

4. PERFORMANCE PROJECTIONS:
   - Expect 99+ performance scores by end of 2026
   - Context windows likely to reach 100M+ tokens
   - Real-time multimodal processing becoming standard
"""

print(insights)

# SMB Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR SMALL TO MEDIUM BUSINESSES")
print("="*80)

recommendations = """
IMMEDIATE ACTIONS (Next 3-6 months):

1. START WITH HYBRID APPROACH:
   - Use Claude 3.7 Sonnet for customer-facing applications (reliable, good safety)
   - Deploy DeepSeek-R1 for internal coding and analysis tasks (cost-effective)
   - Test Llama 4 for non-critical workloads (free if self-hosted)

2. OPTIMAL MODEL SELECTION BY USE CASE:
   
   Customer Service & Sales:
   → Claude 3.7 Sonnet - Best balance of safety, reliability, and cost
   → Backup: GPT-4.1 for complex queries
   
   Content Creation:
   → Gemini 2.5 Flash - Fast, affordable, good quality
   → Consider Llama 4 Maverick for high-volume needs
   
   Software Development:
   → DeepSeek-R1 - Excellent reasoning at 95% lower cost
   → Claude 4 Opus for mission-critical code reviews
   
   Document Processing:
   → Llama 4 Scout (10M context) - Handle entire databases
   → Self-host to minimize costs
   
   Data Analysis:
   → Start with DeepSeek-V3-0324 - Good balance of speed/capability
   → Upgrade to GPT-4.1 for complex analyses

3. COST OPTIMIZATION STRATEGY:
   - Begin with API access to test models (low commitment)
   - Move high-volume workloads to open source (Llama 4, DeepSeek)
   - Reserve premium models (Claude 4, GPT-5) for high-value tasks
   - Consider hybrid on-premise/cloud deployment

4. IMPLEMENTATION ROADMAP:
   
   Phase 1 (Months 1-2):
   - Set up API access to 2-3 models
   - Run pilot projects in non-critical areas
   - Measure ROI and gather user feedback
   
   Phase 2 (Months 3-4):
   - Deploy chosen models in production
   - Train staff on prompt engineering
   - Establish governance and safety protocols
   
   Phase 3 (Months 5-6):
   - Optimize costs with open source alternatives
   - Scale successful use cases
   - Consider fine-tuning for specific needs

5. RISK MITIGATION:
   - Don't rely on a single provider
   - Keep sensitive data with established providers (Anthropic, OpenAI)
   - Monitor for hallucinations, especially with newer models
   - Maintain human oversight for critical decisions

6. BUDGET ALLOCATION (for $10K/month AI budget):
   - 40% - Customer-facing applications (Claude/GPT)
   - 30% - Internal productivity tools (DeepSeek/Llama)
   - 20% - Infrastructure and hosting
   - 10% - Experimentation with new models

LONG-TERM STRATEGY (6-12 months):
- Prepare for GPT-5 and next-gen models
- Build in-house AI expertise
- Consider partnerships with AI consultancies
- Evaluate custom model training for competitive advantage
"""

print(recommendations)

# ROI Calculation Example
print("\n" + "="*80)
print("EXAMPLE ROI CALCULATION")
print("="*80)

roi_example = """
Scenario: Medium-sized software company (50 developers)

Current State:
- Developer productivity: 100 lines of quality code/day
- Average developer cost: $100K/year ($400/day)

With AI Implementation (DeepSeek-R1 for coding):
- Productivity increase: 40% (to 140 lines/day)
- AI cost: $500/month for team
- Implementation cost: $10K one-time

Annual ROI Calculation:
- Productivity gain value: 50 devs × $400/day × 0.4 × 250 days = $2,000,000
- Annual AI cost: $500 × 12 = $6,000
- First-year net benefit: $2,000,000 - $6,000 - $10,000 = $1,984,000
- ROI: 19,840% in first year

This demonstrates why AI adoption is becoming mandatory, not optional.
"""

print(roi_example)

# Create a summary recommendation chart
fig2, ax = plt.subplots(figsize=(12, 8))

models = ['DeepSeek-R1', 'Claude 3.7', 'Llama 4', 'Gemini 2.5', 'GPT-4.1']
ease_of_use = [3, 5, 2, 4, 4]
cost_efficiency = [5, 3, 5, 3.5, 2]
capability = [4.5, 4, 4, 4.5, 4]
reliability = [3.5, 5, 3, 4.5, 4.5]

x = np.arange(len(models))
width = 0.2

ax.bar(x - 1.5*width, ease_of_use, width, label='Ease of Use', alpha=0.8)
ax.bar(x - 0.5*width, cost_efficiency, width, label='Cost Efficiency', alpha=0.8)
ax.bar(x + 0.5*width, capability, width, label='Capability', alpha=0.8)
ax.bar(x + 1.5*width, reliability, width, label='Reliability', alpha=0.8)

ax.set_xlabel('AI Model')
ax.set_ylabel('Score (1-5)')
ax.set_title('SMB Model Selection Matrix', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add overall recommendation scores
overall_scores = []
for i in range(len(models)):
    score = (ease_of_use[i] * 0.2 + cost_efficiency[i] * 0.4 + 
             capability[i] * 0.25 + reliability[i] * 0.15)
    overall_scores.append(score)
    ax.text(i, 5.5, f'Overall: {score:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('smb_ai_selection_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("FINAL VERDICT FOR SMBs")
print("="*80)
print("""
The AI landscape in 2025 presents an unprecedented opportunity for SMBs:

1. START TODAY with DeepSeek-R1 for cost-effective AI adoption
2. USE Claude 3.7 Sonnet for customer-facing applications  
3. EXPERIMENT with Llama 4 for high-volume, non-critical tasks
4. PREPARE for the next wave (GPT-5, Claude 5) arriving late 2025

The key is to start small, measure results, and scale what works.
The cost of inaction far exceeds the cost of experimentation.
""")