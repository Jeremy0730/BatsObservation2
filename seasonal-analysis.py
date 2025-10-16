import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency
import os
import warnings
warnings.filterwarnings('ignore')

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

print("="*80)
print("HIT140 - Investigation B: Seasonal Changes in Bat Behaviour")
print("="*80)
print()

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """
    Load both datasets and perform initial preprocessing.
    Returns: df1 (individual bat landings), df2 (aggregated sessions)
    """
    print("SECTION 1: DATA LOADING AND PREPROCESSING")
    print("-" * 80)
    
    # Load dataset1.csv (individual bat landings)
    print("\n1.1 Loading dataset1.csv (individual bat landings)...")
    df1 = pd.read_csv('dataset1.csv')
    print(f"   - Shape: {df1.shape}")
    print(f"   - Columns: {list(df1.columns)}")
    
    # Load dataset2.csv (30-min aggregated sessions)
    print("\n1.2 Loading dataset2.csv (aggregated sessions)...")
    df2 = pd.read_csv('dataset2.csv')
    print(f"   - Shape: {df2.shape}")
    print(f"   - Columns: {list(df2.columns)}")
    
    # Check for missing values
    print("\n1.3 Checking for missing values...")
    print("\nDataset 1 missing values:")
    missing_df1 = df1.isnull().sum()
    print(missing_df1[missing_df1 > 0] if missing_df1.sum() > 0 else "   No missing values")
    
    print("\nDataset 2 missing values:")
    missing_df2 = df2.isnull().sum()
    print(missing_df2[missing_df2 > 0] if missing_df2.sum() > 0 else "   No missing values")
    
    # Convert datetime columns
    print("\n1.4 Converting datetime columns...")
    try:
        df1['start_time'] = pd.to_datetime(df1['start_time'], format='%d/%m/%Y %H:%M', errors='coerce')
        df1['rat_period_start'] = pd.to_datetime(df1['rat_period_start'], format='%d/%m/%Y %H:%M', errors='coerce')
        df1['rat_period_end'] = pd.to_datetime(df1['rat_period_end'], format='%d/%m/%Y %H:%M', errors='coerce')
        df1['sunset_time'] = pd.to_datetime(df1['sunset_time'], format='%d/%m/%Y %H:%M', errors='coerce')
        print("   - Dataset1 datetime conversion: SUCCESS")
    except Exception as e:
        print(f"   - Dataset1 datetime conversion: ERROR - {e}")
    
    try:
        df2['time'] = pd.to_datetime(df2['time'], format='%d/%m/%Y %H:%M', errors='coerce')
        print("   - Dataset2 datetime conversion: SUCCESS")
    except Exception as e:
        print(f"   - Dataset2 datetime conversion: ERROR - {e}")
    
    # Feature Engineering: Add rat_presence_duration to dataset1
    print("\n1.5 Feature Engineering...")
    df1['rat_presence_duration'] = (df1['rat_period_end'] - df1['rat_period_start']).dt.total_seconds() / 60.0
    print("   - Created 'rat_presence_duration' (minutes) in dataset1")
    
    # Add season labels (assuming: 0=Summer, 1=Autumn, 2=Winter, 3=Spring)
    season_map = {0: 'Summer', 1: 'Autumn', 2: 'Winter', 3: 'Spring'}
    df1['season_name'] = df1['season'].map(season_map)
    print("   - Created 'season_name' labels in dataset1")
    
    # Add month to dataset2 if not properly labeled
    if df2['month'].dtype == 'int64':
        df2['month_extracted'] = df2['time'].dt.month if 'time' in df2.columns else df2['month']
    
    print("\n1.6 Data preprocessing completed successfully!")
    print("="*80)
    print()
    
    return df1, df2


# ============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS BY SEASON
# ============================================================================

def descriptive_statistics_by_season(df1, df2):
    """
    Compute and display descriptive statistics grouped by season.
    """
    print("SECTION 2: DESCRIPTIVE STATISTICS BY SEASON")
    print("-" * 80)
    
    print("\n2.1 Dataset1 - Individual Bat Landings by Season")
    print("-" * 80)
    
    # Group by season
    season_groups = df1.groupby('season_name')
    
    # Key metrics by season
    season_stats = pd.DataFrame({
        'Count': season_groups.size(),
        'Risk_Mean': season_groups['risk'].mean(),
        'Risk_Std': season_groups['risk'].std(),
        'Reward_Mean': season_groups['reward'].mean(),
        'Reward_Std': season_groups['reward'].std(),
        'Time_Delay_Mean': season_groups['bat_landing_to_food'].mean(),
        'Time_Delay_Std': season_groups['bat_landing_to_food'].std(),
        'Rat_Duration_Mean': season_groups['rat_presence_duration'].mean(),
        'Rat_Duration_Std': season_groups['rat_presence_duration'].std()
    })
    
    print(season_stats.round(3))
    
    # Export to CSV
    season_stats.to_csv('figures/season_statistics_dataset1.csv')
    print("\n   - Exported to: figures/season_statistics_dataset1.csv")
    
    print("\n2.2 Dataset2 - Aggregated Sessions by Month")
    print("-" * 80)
    
    # Group dataset2 by month
    month_groups = df2.groupby('month')
    
    month_stats = pd.DataFrame({
        'Count': month_groups.size(),
        'Bat_Landings_Mean': month_groups['bat_landing_number'].mean(),
        'Bat_Landings_Std': month_groups['bat_landing_number'].std(),
        'Food_Availability_Mean': month_groups['food_availability'].mean(),
        'Food_Availability_Std': month_groups['food_availability'].std(),
        'Rat_Minutes_Mean': month_groups['rat_minutes'].mean(),
        'Rat_Minutes_Std': month_groups['rat_minutes'].std(),
        'Rat_Arrivals_Mean': month_groups['rat_arrival_number'].mean(),
        'Rat_Arrivals_Std': month_groups['rat_arrival_number'].std()
    })
    
    print(month_stats.round(3))
    
    # Export to CSV
    month_stats.to_csv('figures/month_statistics_dataset2.csv')
    print("\n   - Exported to: figures/month_statistics_dataset2.csv")
    
    print("\n" + "="*80)
    print()
    
    return season_stats, month_stats


# ============================================================================
# SECTION 3: INVESTIGATION B - STATISTICAL TESTING
# ============================================================================

def investigation_b_statistical_tests(df1, df2):
    """
    Conduct statistical tests to determine if season affects bat behaviour.
    """
    print("SECTION 3: INVESTIGATION B - STATISTICAL TESTING")
    print("-" * 80)
    
    results = {}
    
    # ========== 3.1 Risk-Taking Behaviour by Season ==========
    print("\n3.1 ANOVA/Kruskal-Wallis Test: Risk-Taking by Season")
    print("-" * 80)
    
    # Prepare data by season
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    risk_by_season = [df1[df1['season_name'] == s]['risk'].dropna().values 
                      for s in season_order if s in df1['season_name'].unique()]
    
    # Perform ANOVA (parametric)
    f_stat, p_value_anova = f_oneway(*risk_by_season)
    print(f"\nOne-Way ANOVA Results:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value_anova:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_anova < 0.05 else 'NO - Not significant'}")
    
    # Perform Kruskal-Wallis (non-parametric)
    h_stat, p_value_kw = kruskal(*risk_by_season)
    print(f"\nKruskal-Wallis Test Results:")
    print(f"   H-statistic: {h_stat:.4f}")
    print(f"   p-value: {p_value_kw:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_kw < 0.05 else 'NO - Not significant'}")
    
    results['risk_anova'] = {'F': f_stat, 'p': p_value_anova}
    results['risk_kruskal'] = {'H': h_stat, 'p': p_value_kw}
    
    # ========== 3.2 Reward Rate by Season ==========
    print("\n3.2 ANOVA/Kruskal-Wallis Test: Reward Rate by Season")
    print("-" * 80)
    
    reward_by_season = [df1[df1['season_name'] == s]['reward'].dropna().values 
                        for s in season_order if s in df1['season_name'].unique()]
    
    f_stat, p_value_anova = f_oneway(*reward_by_season)
    print(f"\nOne-Way ANOVA Results:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value_anova:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_anova < 0.05 else 'NO - Not significant'}")
    
    h_stat, p_value_kw = kruskal(*reward_by_season)
    print(f"\nKruskal-Wallis Test Results:")
    print(f"   H-statistic: {h_stat:.4f}")
    print(f"   p-value: {p_value_kw:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_kw < 0.05 else 'NO - Not significant'}")
    
    results['reward_anova'] = {'F': f_stat, 'p': p_value_anova}
    results['reward_kruskal'] = {'H': h_stat, 'p': p_value_kw}
    
    # ========== 3.3 Time Delay by Season ==========
    print("\n3.3 ANOVA/Kruskal-Wallis Test: Time Delay by Season")
    print("-" * 80)
    
    time_by_season = [df1[df1['season_name'] == s]['bat_landing_to_food'].dropna().values 
                      for s in season_order if s in df1['season_name'].unique()]
    
    f_stat, p_value_anova = f_oneway(*time_by_season)
    print(f"\nOne-Way ANOVA Results:")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value_anova:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_anova < 0.05 else 'NO - Not significant'}")
    
    h_stat, p_value_kw = kruskal(*time_by_season)
    print(f"\nKruskal-Wallis Test Results:")
    print(f"   H-statistic: {h_stat:.4f}")
    print(f"   p-value: {p_value_kw:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_kw < 0.05 else 'NO - Not significant'}")
    
    results['time_anova'] = {'F': f_stat, 'p': p_value_anova}
    results['time_kruskal'] = {'H': h_stat, 'p': p_value_kw}
    
    # ========== 3.4 Rat Encounters by Season (Dataset2) ==========
    print("\n3.4 Rat Encounter Frequency by Month (Dataset2)")
    print("-" * 80)
    
    # Map months to seasons (Australian seasons)
    # Summer: Dec(11), Jan(0), Feb(1)
    # Autumn: Mar(2), Apr(3), May(4)
    # Winter: Jun(5), Jul(6), Aug(7)
    # Spring: Sep(8), Oct(9), Nov(10)
    
    def month_to_season(month):
        if month in [11, 0, 1]:
            return 'Summer'
        elif month in [2, 3, 4]:
            return 'Autumn'
        elif month in [5, 6, 7]:
            return 'Winter'
        elif month in [8, 9, 10]:
            return 'Spring'
        return 'Unknown'
    
    df2['season_name'] = df2['month'].apply(month_to_season)
    
    # Test rat arrivals by season
    rat_arrivals_by_season = [df2[df2['season_name'] == s]['rat_arrival_number'].dropna().values 
                              for s in season_order if s in df2['season_name'].unique()]
    
    if len(rat_arrivals_by_season) > 1:
        f_stat, p_value_anova = f_oneway(*rat_arrivals_by_season)
        print(f"\nOne-Way ANOVA Results (Rat Arrivals):")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   p-value: {p_value_anova:.4f}")
        print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_anova < 0.05 else 'NO - Not significant'}")
        
        h_stat, p_value_kw = kruskal(*rat_arrivals_by_season)
        print(f"\nKruskal-Wallis Test Results (Rat Arrivals):")
        print(f"   H-statistic: {h_stat:.4f}")
        print(f"   p-value: {p_value_kw:.4f}")
        print(f"   Significance (alpha=0.05): {'YES - Significant difference' if p_value_kw < 0.05 else 'NO - Not significant'}")
        
        results['rat_arrivals_anova'] = {'F': f_stat, 'p': p_value_anova}
        results['rat_arrivals_kruskal'] = {'H': h_stat, 'p': p_value_kw}
    
    # ========== 3.5 Chi-Square Test: Risk Behaviour Distribution by Season ==========
    print("\n3.5 Chi-Square Test: Risk Behaviour Distribution by Season")
    print("-" * 80)
    
    # Create contingency table
    contingency_table = pd.crosstab(df1['season_name'], df1['risk'])
    print("\nContingency Table:")
    print(contingency_table)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"   Chi-square statistic: {chi2:.4f}")
    print(f"   Degrees of freedom: {dof}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Significance (alpha=0.05): {'YES - Significant association' if p_value < 0.05 else 'NO - Not significant'}")
    
    results['chi_square'] = {'chi2': chi2, 'p': p_value, 'dof': dof}
    
    # Export results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv('figures/investigation_b_test_results.csv')
    print("\n   - Exported test results to: figures/investigation_b_test_results.csv")
    
    print("\n" + "="*80)
    print()
    
    return results


# ============================================================================
# SECTION 4: VISUALIZATIONS
# ============================================================================

def create_seasonal_visualizations(df1, df2):
    """
    Create comprehensive visualizations for Investigation B.
    """
    print("SECTION 4: CREATING VISUALIZATIONS")
    print("-" * 80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # ========== 4.1 Risk-Taking by Season (Boxplot) ==========
    print("\n4.1 Creating boxplot: Risk-Taking by Season...")
    fig, ax = plt.subplots(figsize=(10, 6))
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    sns.boxplot(data=df1, x='season_name', y='risk', order=season_order, palette='Set2', ax=ax)
    ax.set_title('Risk-Taking Behaviour by Season', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Risk Level (0=Avoidance, 1=Risk-Taking)', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/risk_by_season_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/risk_by_season_boxplot.png")
    
    # ========== 4.2 Reward Rate by Season (Boxplot) ==========
    print("\n4.2 Creating boxplot: Reward Rate by Season...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df1, x='season_name', y='reward', order=season_order, palette='Set3', ax=ax)
    ax.set_title('Foraging Success (Reward) by Season', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Reward (0=Failure, 1=Success)', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/reward_by_season_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/reward_by_season_boxplot.png")
    
    # ========== 4.3 Time Delay by Season (Violin Plot) ==========
    print("\n4.3 Creating violin plot: Time Delay by Season...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df1, x='season_name', y='bat_landing_to_food', order=season_order, 
                   palette='muted', ax=ax, inner='quartile')
    ax.set_title('Bat Landing to Food Time Delay by Season', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Time Delay (seconds)', fontsize=12)
    ax.set_ylim(0, 100)  # Limit y-axis for better visibility
    plt.tight_layout()
    plt.savefig('figures/time_delay_by_season_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/time_delay_by_season_violin.png")
    
    # ========== 4.4 Bar Chart: Mean Risk and Reward by Season ==========
    print("\n4.4 Creating bar chart: Mean Risk and Reward by Season...")
    season_summary = df1.groupby('season_name')[['risk', 'reward']].mean().reindex(season_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(season_order))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, season_summary['risk'], width, label='Risk-Taking', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, season_summary['reward'], width, label='Reward', color='skyblue', alpha=0.8)
    
    ax.set_title('Mean Risk-Taking and Reward Rate by Season', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Mean Rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(season_order)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/risk_reward_by_season_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/risk_reward_by_season_bar.png")
    
    # ========== 4.5 Rat Arrivals by Season (Dataset2) ==========
    print("\n4.5 Creating line chart: Rat Arrivals by Season...")
    season_rat_summary = df2.groupby('season_name')[['rat_arrival_number', 'rat_minutes']].mean().reindex(season_order)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rat arrival numbers
    ax1.plot(season_order, season_rat_summary['rat_arrival_number'], marker='o', 
             linewidth=2, markersize=8, color='darkred', label='Rat Arrivals')
    ax1.set_title('Mean Rat Arrival Number by Season', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Season', fontsize=11)
    ax1.set_ylabel('Mean Rat Arrival Number', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Rat minutes
    ax2.plot(season_order, season_rat_summary['rat_minutes'], marker='s', 
             linewidth=2, markersize=8, color='darkgreen', label='Rat Minutes')
    ax2.set_title('Mean Rat Presence Duration by Season', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=11)
    ax2.set_ylabel('Mean Rat Minutes', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/rat_presence_by_season_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/rat_presence_by_season_line.png")
    
    # ========== 4.6 Heatmap: Correlation Matrix by Season ==========
    print("\n4.6 Creating heatmap: Correlation Matrix...")
    correlation_vars = ['risk', 'reward', 'bat_landing_to_food', 'hours_after_sunset', 'rat_presence_duration']
    corr_matrix = df1[correlation_vars].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Key Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/correlation_heatmap.png")
    
    # ========== 4.7 Comprehensive Dashboard ==========
    print("\n4.7 Creating comprehensive dashboard...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Risk by Season
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df1, x='season_name', y='risk', order=season_order, palette='Set2', ax=ax1)
    ax1.set_title('(A) Risk-Taking by Season', fontweight='bold')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Risk Level')
    
    # Plot 2: Reward by Season
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=df1, x='season_name', y='reward', order=season_order, palette='Set3', ax=ax2)
    ax2.set_title('(B) Reward Rate by Season', fontweight='bold')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Reward')
    
    # Plot 3: Time Delay Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    for season in season_order:
        data = df1[df1['season_name'] == season]['bat_landing_to_food']
        ax3.hist(data, alpha=0.5, label=season, bins=30)
    ax3.set_title('(C) Time Delay Distribution by Season', fontweight='bold')
    ax3.set_xlabel('Time Delay (seconds)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_xlim(0, 50)
    
    # Plot 4: Risk vs Reward Scatter
    ax4 = fig.add_subplot(gs[1, 1])
    for season in season_order:
        season_data = df1[df1['season_name'] == season]
        ax4.scatter(season_data['risk'], season_data['reward'], alpha=0.4, label=season, s=20)
    ax4.set_title('(D) Risk vs Reward by Season', fontweight='bold')
    ax4.set_xlabel('Risk Level')
    ax4.set_ylabel('Reward')
    ax4.legend()
    
    # Plot 5: Bat Landings and Food Availability (Dataset2)
    ax5 = fig.add_subplot(gs[2, 0])
    season_summary_df2 = df2.groupby('season_name')[['bat_landing_number', 'food_availability']].mean().reindex(season_order)
    x_pos = np.arange(len(season_order))
    width = 0.35
    ax5.bar(x_pos - width/2, season_summary_df2['bat_landing_number'], width, label='Bat Landings', alpha=0.7)
    ax5_twin = ax5.twinx()
    ax5_twin.bar(x_pos + width/2, season_summary_df2['food_availability'], width, 
                 label='Food Availability', alpha=0.7, color='orange')
    ax5.set_title('(E) Bat Activity & Food Availability by Season', fontweight='bold')
    ax5.set_xlabel('Season')
    ax5.set_ylabel('Mean Bat Landings')
    ax5_twin.set_ylabel('Mean Food Availability')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(season_order)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    
    # Plot 6: Summary Statistics Table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_table = df1.groupby('season_name')[['risk', 'reward']].agg(['mean', 'std']).round(3)
    table_data = []
    table_data.append(['Season', 'Risk Mean', 'Risk Std', 'Reward Mean', 'Reward Std'])
    for season in season_order:
        if season in summary_table.index:
            row = summary_table.loc[season]
            table_data.append([season, f"{row[('risk', 'mean')]:.3f}", f"{row[('risk', 'std')]:.3f}",
                             f"{row[('reward', 'mean')]:.3f}", f"{row[('reward', 'std')]:.3f}"])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('(F) Summary Statistics by Season', fontweight='bold', pad=20)
    
    plt.suptitle('Investigation B: Seasonal Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('figures/investigation_b_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Saved: figures/investigation_b_dashboard.png")
    
    print("\n" + "="*80)
    print()


# ============================================================================
# SECTION 5: CONCLUSIONS AND INTERPRETATION
# ============================================================================

def generate_conclusions(results, season_stats):
    """
    Generate final conclusions for Investigation B.
    """
    print("SECTION 5: INVESTIGATION B - CONCLUSIONS")
    print("="*80)
    
    print("\nResearch Question: Do bat behaviours change following seasonal changes?")
    print("-" * 80)
    
    print("\nKey Findings:")
    print("-" * 80)
    
    # Risk-taking conclusion
    if results['risk_anova']['p'] < 0.05:
        print("\n1. RISK-TAKING BEHAVIOUR BY SEASON:")
        print(f"   - ANOVA p-value: {results['risk_anova']['p']:.4f} (SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['risk_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There IS a statistically significant difference in risk-taking")
        print("     behaviour across seasons.")
        print("\n   Season-specific patterns:")
        for season in season_stats.index:
            print(f"     • {season}: Mean risk = {season_stats.loc[season, 'Risk_Mean']:.3f} "
                  f"(SD = {season_stats.loc[season, 'Risk_Std']:.3f})")
    else:
        print("\n1. RISK-TAKING BEHAVIOUR BY SEASON:")
        print(f"   - ANOVA p-value: {results['risk_anova']['p']:.4f} (NOT SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['risk_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There is NO statistically significant difference in risk-taking")
        print("     behaviour across seasons.")
    
    # Reward rate conclusion
    if results['reward_anova']['p'] < 0.05:
        print("\n2. FORAGING SUCCESS (REWARD) BY SEASON:")
        print(f"   - ANOVA p-value: {results['reward_anova']['p']:.4f} (SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['reward_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There IS a statistically significant difference in foraging")
        print("     success across seasons.")
    else:
        print("\n2. FORAGING SUCCESS (REWARD) BY SEASON:")
        print(f"   - ANOVA p-value: {results['reward_anova']['p']:.4f} (NOT SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['reward_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There is NO statistically significant difference in foraging")
        print("     success across seasons.")
    
    # Time delay conclusion
    if results['time_anova']['p'] < 0.05:
        print("\n3. TIME DELAY BY SEASON:")
        print(f"   - ANOVA p-value: {results['time_anova']['p']:.4f} (SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['time_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There IS a statistically significant difference in foraging")
        print("     time delay across seasons.")
    else:
        print("\n3. TIME DELAY BY SEASON:")
        print(f"   - ANOVA p-value: {results['time_anova']['p']:.4f} (NOT SIGNIFICANT)")
        print(f"   - Kruskal-Wallis p-value: {results['time_kruskal']['p']:.4f}")
        print("   - CONCLUSION: There is NO statistically significant difference in foraging")
        print("     time delay across seasons.")
    
    # Rat encounters conclusion
    if 'rat_arrivals_anova' in results:
        if results['rat_arrivals_anova']['p'] < 0.05:
            print("\n4. RAT ENCOUNTER FREQUENCY BY SEASON:")
            print(f"   - ANOVA p-value: {results['rat_arrivals_anova']['p']:.4f} (SIGNIFICANT)")
            print(f"   - Kruskal-Wallis p-value: {results['rat_arrivals_kruskal']['p']:.4f}")
            print("   - CONCLUSION: There IS a statistically significant difference in rat")
            print("     encounter frequency across seasons.")
        else:
            print("\n4. RAT ENCOUNTER FREQUENCY BY SEASON:")
            print(f"   - ANOVA p-value: {results['rat_arrivals_anova']['p']:.4f} (NOT SIGNIFICANT)")
            print(f"   - Kruskal-Wallis p-value: {results['rat_arrivals_kruskal']['p']:.4f}")
            print("   - CONCLUSION: There is NO statistically significant difference in rat")
            print("     encounter frequency across seasons.")
    
    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION FOR INVESTIGATION B:")
    print("="*80)
    
    significant_tests = sum([
        results['risk_anova']['p'] < 0.05,
        results['reward_anova']['p'] < 0.05,
        results['time_anova']['p'] < 0.05,
        results.get('rat_arrivals_anova', {}).get('p', 1.0) < 0.05
    ])
    
    if significant_tests >= 2:
        print("\nAnswer: YES - Bat behaviours DO change with seasonal changes.")
        print("\nEvidence:")
        print(f"  • {significant_tests} out of 4 key variables showed statistically significant")
        print("    seasonal differences (p < 0.05)")
        print("\nInterpretation:")
        print("  The hypothesis is SUPPORTED. Seasonal variations in food availability and")
        print("  environmental conditions appear to influence bat foraging behaviours and")
        print("  their responses to rat presence.")
    elif significant_tests == 1:
        print("\nAnswer: PARTIALLY SUPPORTED - Some bat behaviours change with seasons.")
        print("\nEvidence:")
        print("  • 1 out of 4 key variables showed statistically significant seasonal differences")
        print("\nInterpretation:")
        print("  The hypothesis is PARTIALLY SUPPORTED. While some seasonal effects are")
        print("  observable, the overall pattern is not consistently strong across all")
        print("  behavioural metrics.")
    else:
        print("\nAnswer: NO - Bat behaviours do NOT significantly change with seasonal changes.")
        print("\nEvidence:")
        print("  • None of the key variables showed statistically significant seasonal differences")
        print("    (all p-values > 0.05)")
        print("\nInterpretation:")
        print("  The hypothesis is NOT SUPPORTED. Despite observed descriptive differences,")
        print("  statistical tests indicate that seasonal variations do not significantly")
        print("  influence bat foraging behaviours in this dataset.")
    
    print("\n" + "="*80)
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for Investigation B analysis.
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  HIT140 - FOUNDATIONS OF DATA SCIENCE".center(78) + "*")
    print("*" + "  Assessment 3: Group Project - Investigation B".center(78) + "*")
    print("*" + "  Bat vs. Rat – The Forage Files".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    # Execute analysis pipeline
    df1, df2 = load_and_preprocess_data()
    season_stats, month_stats = descriptive_statistics_by_season(df1, df2)
    results = investigation_b_statistical_tests(df1, df2)
    create_seasonal_visualizations(df1, df2)
    generate_conclusions(results, season_stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll results, visualizations, and summary tables have been saved to the")
    print("'figures/' directory. You can now use these outputs for your group report.")
    print("\nGenerated Files:")
    print("  • figures/season_statistics_dataset1.csv")
    print("  • figures/month_statistics_dataset2.csv")
    print("  • figures/investigation_b_test_results.csv")
    print("  • figures/risk_by_season_boxplot.png")
    print("  • figures/reward_by_season_boxplot.png")
    print("  • figures/time_delay_by_season_violin.png")
    print("  • figures/risk_reward_by_season_bar.png")
    print("  • figures/rat_presence_by_season_line.png")
    print("  • figures/correlation_heatmap.png")
    print("  • figures/investigation_b_dashboard.png")
    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    main()

