import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns

def render_pitcher_analysis(pitcher_data, name):
    plt.style.use('seaborn-darkgrid') 
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--' 
    plt.rcParams['grid.color'] = '#B0B0B0' 

    fig = plt.figure(figsize=(28, 24)) 
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.15, 1, 1.5, 0.75]) 
    fig.patch.set_linewidth(3)  
    fig.patch.set_edgecolor('#404040')

    pitch_color_mapping = {
        'CH': '#ff7f0e', 'SL': '#ff9896', 'CU': '#d62728', 'FS': '#bcbd22',
        'FF': '#1f77b4', 'SC': '#2ca02c', 'SI': '#17becf', 'SV': '#ffbb78',
        'FO': '#7f7f7f', 'KC': '#8c564b', 'ST': '#c5b0d5', 'FC': '#9467bd',
        'CS': '#e377c2'
    }

    ax_overview = fig.add_subplot(gs[0, :])
    display_count_statistics(pitcher_data, ax_overview, fontsize=25)

    ax_left = fig.add_subplot(gs[1, 0])
    plot_pitch_locations(pitcher_data[pitcher_data['BatterSide'] == 'L'], ax_left, 'LHH', pitch_color_mapping)

    ax_right = fig.add_subplot(gs[1, 1])
    plot_pitch_locations(pitcher_data[pitcher_data['BatterSide'] == 'R'], ax_right, 'RHH', pitch_color_mapping)

    ax_break = fig.add_subplot(gs[1, 2])
    plot_pitch_break_analysis(pitcher_data, ax_break, pitch_color_mapping)

    ax_pitches = fig.add_subplot(gs[2, :])
    display_pitch_statistics(pitcher_data, ax_pitches, fontsize=25)

    pitcher_hand = pitcher_data['PitcherHand'].iloc[0]
    handedness_label = "LHP" if pitcher_hand == "L" else "RHP"
    fig.suptitle(f"{name} ({handedness_label})", fontsize=35, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)
    plt.show()

def plot_pitch_locations(data, ax, title, pitch_color_map):
    avg_locations = data.groupby('PitchType').agg(
        avg_x=('TrajectoryLocationX', 'mean'),
        avg_z=('TrajectoryLocationZ', 'mean')
    ).reset_index()

    for _, row in avg_locations.iterrows():
        ax.scatter(row['avg_x'], row['avg_z'], 
                   label=row['PitchType'], 
                   alpha=0.7, 
                   s=400, 
                   color=pitch_color_map.get(row['PitchType'], 'gray'))

    zone_bottom = data['StrikeZoneBottom'].mean()
    zone_top = data['StrikeZoneTop'].mean()
    zone_width = 1.4166
    zone_height = zone_top - zone_bottom
    zone_center = 0
    strike_zone = Rectangle((zone_center - zone_width / 2, zone_bottom), zone_width, zone_height, 
                            fill=False, color='k', linewidth=2)
    ax.add_patch(strike_zone)

    plate_width = zone_width
    plate = plt.Polygon([(-plate_width / 2, 0), (0, 0), (plate_width / 2, 0)], color='k')
    ax.add_patch(plate)

    padding_x = 0.5
    padding_y = 0.5
    ax.set_xlim((zone_center - zone_width / 2) - padding_x, (zone_center + zone_width / 2) + padding_x)
    ax.set_ylim(zone_bottom - padding_y, zone_top + padding_y)

    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('', fontsize=18)
    ax.set_title(f'{title}', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Pitch Type", loc="upper right", fontsize=8, title_fontsize='12')
    ax.invert_xaxis()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect(1.02)

def plot_pitch_break_analysis(data, ax, pitch_color_map):
    data['TrajectoryHorizontalBreak'] *= 12
    data['TrajectoryVerticalBreakInduced'] *= 12
    
    for pitch_type in data['PitchType'].unique():
        pitch_data = data[data['PitchType'] == pitch_type]
        ax.scatter(pitch_data['TrajectoryHorizontalBreak'], 
                   pitch_data['TrajectoryVerticalBreakInduced'], 
                   label=pitch_type, 
                   alpha=0.7, 
                   s=70, 
                   color=pitch_color_map.get(pitch_type, 'gray'))
    
    ax.set_xlabel('Horizontal Break (in)', fontsize=26)
    ax.set_ylabel('Vertical Break (in)', fontsize=26)
    ax.set_title('Pitch Break', fontsize=28)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Pitch Type", loc="upper right", fontsize=14, title_fontsize='16')

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    max_horz = data['TrajectoryHorizontalBreak'].max() * 1.1
    min_horz = data['TrajectoryHorizontalBreak'].min() * 1.1
    max_vert = data['TrajectoryVerticalBreakInduced'].max() * 1.1
    min_vert = data['TrajectoryVerticalBreakInduced'].min() * 1.1

    ax.set_xlim(min_horz, max_horz)
    ax.set_ylim(min_vert, max_vert)

    ax.tick_params(axis='both', which='major', labelsize=14)

def display_count_statistics(pitcher_data, ax, fontsize=25):
    batters_faced = pitcher_data['AtBatNumber'].nunique()
    strikeouts = (pitcher_data['PitchCall'] == 'strikeout').sum()
    walks = (pitcher_data['PitchCall'] == 'walk').sum()
    singles = (pitcher_data['PitchCall'] == 'single').sum()
    doubles = (pitcher_data['PitchCall'] == 'double').sum()
    triples = (pitcher_data['PitchCall'] == 'triple').sum()
    home_runs = (pitcher_data['PitchCall'] == 'home_run').sum()

    hits = singles + doubles + triples + home_runs
    at_bats = batters_faced - walks - (pitcher_data['PitchCall'] == 'hit_by_pitch').sum()

    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

    valid_pitches = pitcher_data['PitchType'].notna().sum()
    
    outs = (
        strikeouts +
        (pitcher_data['PitchCall'] == 'field_out').sum() +
        (pitcher_data['PitchCall'] == 'sac_bunt').sum() +
        (pitcher_data['PitchCall'] == 'force_out').sum() +
        2 * (pitcher_data['PitchCall'] == 'grounded_into_double_play').sum()
    )
    
    innings_pitched = f"{outs // 3}"
    if outs % 3 == 1:
        innings_pitched += " 1/3"
    elif outs % 3 == 2:
        innings_pitched += " 2/3"
    
    WHIP = float(walks+hits)/float(innings_pitched)
    count_stats = {
        'Pitches': valid_pitches,
        'PA': batters_faced,
        'Ks': strikeouts,
        'BBs': walks,
        'HRs': home_runs,
        'Hits': hits,
        'WHIP': WHIP,
        'Opp SLG': f"{(total_bases / at_bats):.3f}" if at_bats > 0 else "0.000"
    }
    
    count_stats['IP'] = innings_pitched


    

    df_counts = pd.DataFrame([count_stats])
    table = ax.table(cellText=df_counts.values, colLabels=df_counts.columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(True)
    table.scale(1, 4)
    ax.axis('off')

def display_pitch_statistics(pitcher_data, ax, fontsize=25):
    pitcher_data['true_spin'] = np.sqrt(
        pitcher_data['SpinVectorX']**2 + 
        pitcher_data['SpinVectorZ']**2
    )
    pitcher_data['spin_efficiency'] = (pitcher_data['true_spin'] / pitcher_data['ReleaseSpinRate']) * 100
    pitcher_data['spin_efficiency'] = np.clip(pitcher_data['spin_efficiency'], 0, 100)

    at_bat_outcomes = ['single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 'walk']
    pitch_counts = pitcher_data['PitchType'].value_counts()

    pitch_stats = pitcher_data.groupby('PitchType').agg(
        count=('PitchType', 'size'),
        avg_velocity=('ReleaseSpeed', 'mean'),
        avg_spin_rate=('ReleaseSpinRate', 'mean'),
        avg_spin_efficiency=('spin_efficiency', 'mean'),
        avg_pitching_hand=('PitcherHand', 'first'),
        avg_vertical_break=('TrajectoryVerticalBreakInduced', 'mean'),
        avg_horizontal_break=('TrajectoryHorizontalBreak', 'mean'),
        whiff_percentage=('PitchCall', lambda x: sum(x == 'swinging_strike') / len(x))
    ).reset_index()

    pitch_stats['spin_efficiency'] = pitch_stats['avg_spin_efficiency'].round(1)
    pitch_stats['avg_velocity'] = pitch_stats['avg_velocity'].round(1)
    pitch_stats['avg_spin_rate'] = pitch_stats['avg_spin_rate'].round(1)

    pitch_stats['avg_batter_hand'] = pitcher_data['BatterSide'].value_counts()
    pitch_stats['avg_batter_hand'] = pitch_stats['avg_batter_hand'].get('R', 0)

    pitch_stats['whiff_rate'] = pitch_stats['whiff_percentage'].apply(lambda x: f"{x:.1%}")

    pitch_stats = pitch_stats[['PitchType', 'count', 'avg_velocity', 'avg_spin_rate', 'spin_efficiency', 'avg_vertical_break', 'avg_horizontal_break', 'whiff_rate']]

    columns = pitch_stats.columns
    table = ax.table(cellText=pitch_stats.values, colLabels=pitch_stats.columns, 
                     cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.8)
    ax.axis('off')

def display_batter_hand_analysis(pitcher_data, ax, title):
    hand_counts = pitcher_data['BatterSide'].value_counts()
    ax.pie(hand_counts, labels=hand_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)

if __name__ == '__main__':
    pitch_data = pd.read_csv("AnalyticsQuestionnairePitchData.csv")
    
    for uniq in pitch_data['PitcherId'].unique():
        render_pitcher_analysis(pitch_data[pitch_data['PitcherId'] == uniq], f"Pitcher {uniq}")
