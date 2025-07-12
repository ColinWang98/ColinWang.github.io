import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import defaultdict
import re
import matplotlib.cm as cm

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'analysis_results')
os.makedirs(RESULT_DIR, exist_ok=True)

GROUPS = [d for d in os.listdir('.') if d.startswith('P') and os.path.isdir(d)]

# 文件名解析
file_pat = re.compile(r'force_data_(P\d+)([EI])([NF])_.*\.xlsx')

# 阶段定义
PHASES = ['吸气', '吸后屏气', '呼气', '呼后屏气']
PHASES_EN = ['Inhale', 'Hold-in', 'Exhale', 'Hold-out']

# 力值差分阈值（可调）
DIFF_TH = 0.05

# 单文件分析
def analyze_breathing_file(filepath):
    df = pd.read_excel(filepath, usecols=[0,1], header=None, skiprows=1)
    df.columns = ['Timestamp', 'Force']
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    force = df['Force'].values
    timestamps = df['Timestamp']
    # 平滑
    window = max(5, int(len(force)//30)|1)
    force_smooth = savgol_filter(force, window_length=window, polyorder=2) if window < len(force) else force
    force_smooth = np.asarray(force_smooth)
    # 力值差分
    diffs = np.diff(force_smooth, prepend=force_smooth[0])
    # 阈值判定
    states = []
    for d in diffs:
        if d > DIFF_TH:
            states.append('吸气')
        elif d < -DIFF_TH:
            states.append('呼气')
        else:
            states.append('HOLD')
    # 区分屏气
    processed = []
    last = '吸气'
    for st in states:
        if st == 'HOLD':
            processed.append('吸后屏气' if last=='吸气' else '呼后屏气')
        else:
            processed.append(st)
            last = st
    # 合并阶段
    phase_list = []
    cur = processed[0]
    start = 0
    for i, st in enumerate(processed[1:], 1):
        if st != cur or i == len(processed)-1:
            end = i if st != cur else i+1
            duration = (timestamps.iloc[end-1] - timestamps.iloc[start]).total_seconds()
            if duration > 0.5: # 至少0.5秒
                phase_list.append({'state': cur, 'start': start, 'end': end-1, 'duration': duration})
            cur = st
            start = i
    # 按顺序找完整循环
    cycles = []
    i = 0
    while i < len(phase_list)-3:
        seq = [phase_list[j]['state'] for j in range(i,i+4)]
        if seq == PHASES:
            cycles.append(phase_list[i:i+4])
            i += 4
        else:
            i += 1
    # 统计
    phase_durations = defaultdict(list)
    for cyc in cycles:
        for idx, ph in enumerate(PHASES):
            phase_durations[ph].append(cyc[idx]['duration'])
    total_time = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    freq = len(cycles) / (total_time/60) if total_time>0 else 0
    return phase_durations, freq, len(cycles), df

# 为每个人生成四个文件的合并图表
def generate_person_combined_plots():
    """为每个人生成包含四个文件呼吸力值数据的合并图表"""
    person_plots_dir = os.path.join(RESULT_DIR, 'person_combined_plots')
    os.makedirs(person_plots_dir, exist_ok=True)
    
    # 按人员分组收集数据
    person_files = defaultdict(list)
    for group in GROUPS:
        group_path = os.path.join('.', group)
        for fname in os.listdir(group_path):
            m = file_pat.match(fname)
            if not m:
                continue
            group_id, ie, nf = m.group(1), m.group(2), m.group(3)
            person_files[group_id].append((fname, ie, nf, os.path.join(group_path, fname)))
    
    # 为每个人生成图表
    for person_id, files in person_files.items():
        if len(files) != 4:  # 确保每个人有4个文件
            print(f"警告: {person_id} 只有 {len(files)} 个文件，跳过")
            continue
            
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{person_id} 呼吸力值数据 - 四个文件对比', fontsize=16, fontweight='bold')
        
        # 颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        file_types = ['IN', 'EN', 'IF', 'EF']
        
        for idx, (fname, ie, nf, fpath) in enumerate(files):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            try:
                # 读取数据
                df = pd.read_excel(fpath, usecols=[0,1], header=None, skiprows=1)
                df.columns = ['Timestamp', 'Force']
                if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                # 数据平滑
                force = df['Force'].values
                window = max(5, int(len(force)//30)|1)
                force_smooth = savgol_filter(force, window_length=window, polyorder=2) if window < len(force) else force
                
                # 转换为相对时间（秒）
                start_time = df['Timestamp'].iloc[0]
                time_seconds = (df['Timestamp'] - start_time).dt.total_seconds()
                
                # 绘制原始数据和平滑数据
                ax.plot(time_seconds, force, alpha=0.3, color='lightgray', label='原始数据', linewidth=0.5)
                ax.plot(time_seconds, force_smooth, color=colors[idx], label='平滑数据', linewidth=1.5)
                
                # 添加阶段标注
                _, _, _, phase_data = analyze_breathing_file(fpath)
                if phase_data is not None:
                    # 这里可以添加阶段标注，但为了简洁暂时省略
                    pass
                
                ax.set_xlabel('时间 (秒)', fontsize=12)
                ax.set_ylabel('力值', fontsize=12)
                ax.set_title(f'{file_types[idx]} ({ie}{nf})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 设置y轴范围一致
                ax.set_ylim(bottom=0)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'文件读取错误:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=12, color='red')
                ax.set_title(f'{file_types[idx]} ({ie}{nf}) - 错误', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        output_path = os.path.join(person_plots_dir, f'{person_id}_combined_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成 {person_id} 的合并图表: {output_path}")
    
    print(f"\n所有个人合并图表已保存到: {person_plots_dir}")

def main():
    summary = []
    all_phase = defaultdict(list)
    all_freq = []
    label_map = {}
    for group in GROUPS:
        group_path = os.path.join('.', group)
        for fname in os.listdir(group_path):
            m = file_pat.match(fname)
            if not m:
                continue
            group_id, ie, nf = m.group(1), m.group(2), m.group(3)
            label = f'{group_id}-{ie}{nf}'
            fpath = os.path.join(group_path, fname)
            phase_durations, freq, cycles = analyze_breathing_file(fpath)[:3]
            row = {'group': group_id, 'IE': ie, 'NF': nf, 'file': fname, 'cycles': cycles, 'freq': freq}
            for ph in PHASES:
                vals = phase_durations[ph]
                row[f'{ph}_mean'] = np.mean(vals) if vals else 0
                row[f'{ph}_std'] = np.std(vals) if vals else 0
                row[f'{ph}_cv'] = np.std(vals)/np.mean(vals) if vals and np.mean(vals)>0 else 0
                all_phase[(label, ph)].extend(vals)
            all_freq.append((label, freq))
            label_map[label] = (group_id, ie, nf)
            summary.append(row)
    # 保存summary
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(RESULT_DIR, 'summary.csv'), index=False)
    # 汇总雷达图（各组-类型的四阶段CV）
    radar_data = {}
    for label in sorted(label_map):
        cv_list = []
        for ph in PHASES:
            vals = all_phase[(label, ph)]
            cv = np.std(vals)/np.mean(vals) if vals and np.mean(vals)>0 else 0
            cv_list.append(cv)
        radar_data[label] = cv_list
    fig = plt.figure(figsize=(10,8))
    angles = np.linspace(0, 2*np.pi, len(PHASES_EN), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    color_map = plt.get_cmap('Set2')
    for idx, (label, cvs) in enumerate(radar_data.items()):
        vals = np.array(cvs)
        vals = np.concatenate((vals, [vals[0]]))
        plt.polar(angles, vals, label=label, color=color_map(idx%8), linewidth=2)
        plt.fill(angles, vals, alpha=0.15, color=color_map(idx%8))
    plt.thetagrids(np.degrees(angles[:-1]), PHASES_EN, fontsize=13)
    plt.title('Breathing Phase CV Comparison (All Groups)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=11)
    plt.tight_layout(rect=(0,0,0.85,1))
    plt.savefig(os.path.join(RESULT_DIR, 'overall_stability_radar.png'), dpi=200)
    plt.close()
    # 各阶段箱线图（所有组-类型汇总）
    plt.figure(figsize=(max(12, len(label_map)*2), 7))
    boxdata = [all_phase[(label, ph)] for ph in PHASES for label in sorted(label_map)]
    labels = [f'{ph_en}\n{label}' for ph_en in PHASES_EN for label in sorted(label_map)]
    box = plt.boxplot(boxdata, medianprops=dict(color='red'), patch_artist=True)
    n_boxes = len(box['boxes'])
    color_map = plt.get_cmap('tab20', n_boxes)
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(color_map(i))
    plt.xticks(range(1,len(labels)+1), labels, rotation=45, ha='right', fontsize=11)
    plt.title('Boxplot of Phase Durations (All Groups)', fontsize=16)
    plt.ylabel('Duration (s)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'all_phase_boxplot.png'), dpi=200)
    plt.close()
    # 呼吸频率分布
    plt.figure(figsize=(max(12, len(label_map)*1.2), 6))
    freq_labels, freq_vals = zip(*all_freq)
    color_map = plt.get_cmap('tab20', len(freq_labels))
    bar_colors = [color_map(i) for i in range(len(freq_labels))]
    plt.bar(freq_labels, freq_vals, color=bar_colors)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylabel('Breathing Frequency (cycles/min)', fontsize=13)
    plt.title('Breathing Frequency Distribution (All Groups)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'all_freq_bar.png'), dpi=200)
    plt.close()
    # 呼吸阶段平均时长热力图
    heatmap_data = []
    heatmap_labels = []
    for label in sorted(label_map):
        row = []
        for ph in PHASES:
            vals = all_phase[(label, ph)]
            row.append(np.mean(vals) if vals else 0)
        heatmap_data.append(row)
        heatmap_labels.append(label)
    heatmap_data = np.array(heatmap_data)
    plt.figure(figsize=(max(10, len(heatmap_labels)*0.7), 6))
    im = plt.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Mean Duration (s)', shrink=0.8)
    plt.xticks(range(len(heatmap_labels)), heatmap_labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(range(len(PHASES_EN)), PHASES_EN, fontsize=13)
    plt.title('Heatmap of Mean Phase Durations (All Groups)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'phase_duration_heatmap.png'), dpi=200)
    plt.close()
    # --- Group (IN, EN, IF, EF) overall summary ---
    group_types = ['IN', 'EN', 'IF', 'EF']
    group_map = {'IN': ('I','N'), 'EN': ('E','N'), 'IF': ('I','F'), 'EF': ('E','F')}
    group_summary = []
    for gtype in group_types:
        ie, nf = group_map[gtype]
        # 找到所有该类型的label
        labels = [label for label in label_map if label.endswith(f'{ie}{nf}')]
        # 汇总所有阶段时长和频率
        phase_vals = {ph: [] for ph in PHASES}
        freqs = []
        for label in labels:
            for ph in PHASES:
                phase_vals[ph].extend(all_phase[(label, ph)])
            for row in summary:
                if row['group']+ '-' + row['IE'] + row['NF'] == label:
                    freqs.append(row['freq'])
        row = {'group_type': gtype}
        for idx, ph in enumerate(PHASES):
            vals = phase_vals[ph]
            row[f'{PHASES_EN[idx]}_mean'] = str(float(np.mean(vals))) if vals else '0.0'
            row[f'{PHASES_EN[idx]}_std'] = str(float(np.std(vals))) if vals else '0.0'
            row[f'{PHASES_EN[idx]}_cv'] = str(float(np.std(vals)/np.mean(vals))) if vals and np.mean(vals)>0 else '0.0'
        row['freq_mean'] = str(float(np.mean(freqs))) if freqs else '0.0'
        row['freq_std'] = str(float(np.std(freqs))) if freqs else '0.0'
        group_summary.append(row)
    df_group = pd.DataFrame(group_summary)
    df_group.to_csv(os.path.join(RESULT_DIR, 'group_summary.csv'), index=False)
    # 输出终端总结
    print('\n===== Group Type Overall Summary =====')
    for row in group_summary:
        print(f"{row['group_type']}: ", end='')
        for idx, ph in enumerate(PHASES_EN):
            print(f"{ph} mean={row[f'{ph}_mean']}, CV={row[f'{ph}_cv']}; ", end='')
        print(f"Freq mean={row['freq_mean']}/min")
    print(f'分析完成，结果保存在 {RESULT_DIR}')
    # --- Group comparison plots ---
    group_types = ['IN', 'EN', 'IF', 'EF']
    group_colors = [cm.get_cmap('Set2', 4)(i) for i in range(4)]
    df_group = pd.read_csv(os.path.join(RESULT_DIR, 'group_summary.csv'))
    # 1. 各阶段均值分组柱状图
    plt.figure(figsize=(10,6))
    width = 0.18
    x = np.arange(len(PHASES_EN))
    for i, g in enumerate(group_types):
        means = [float(df_group.loc[df_group['group_type']==g, f'{ph}_mean'].values[0]) for ph in PHASES_EN]
        plt.bar(x + i*width, means, width, label=g, color=group_colors[i])
    plt.xticks(x + 1.5*width, PHASES_EN, fontsize=13)
    plt.ylabel('Mean Duration (s)', fontsize=13)
    plt.title('Group Comparison: Mean Phase Durations', fontsize=16)
    plt.legend(title='Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'group_phase_mean_bar.png'), dpi=200)
    plt.close()
    # 2. 各阶段CV分组柱状图
    plt.figure(figsize=(10,6))
    for i, g in enumerate(group_types):
        cvs = [float(df_group.loc[df_group['group_type']==g, f'{ph}_cv'].values[0]) for ph in PHASES_EN]
        plt.bar(x + i*width, cvs, width, label=g, color=group_colors[i])
    plt.xticks(x + 1.5*width, PHASES_EN, fontsize=13)
    plt.ylabel('CV', fontsize=13)
    plt.title('Group Comparison: Phase Duration CV', fontsize=16)
    plt.legend(title='Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'group_phase_cv_bar.png'), dpi=200)
    plt.close()
    # 3. 呼吸频率均值分组柱状图
    plt.figure(figsize=(8,6))
    freq_means = [float(df_group.loc[df_group['group_type']==g, 'freq_mean'].values[0]) for g in group_types]
    plt.bar(group_types, freq_means, color=group_colors)
    plt.ylabel('Mean Breathing Frequency (cycles/min)', fontsize=13)
    plt.title('Group Comparison: Mean Breathing Frequency', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'group_freq_bar.png'), dpi=200)
    plt.close()
    
    # 生成每个人的合并图表
    print("\n正在生成每个人的合并图表...")
    generate_person_combined_plots()

if __name__ == '__main__':
    main() 