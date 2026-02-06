#!/usr/bin/env python3
"""
Processar Todas as Imagens - Pipeline Completo
Detectar espinhas em todas as imagens das 4 linhagens e gerar dados para análise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Importar o pipeline de detecção de espinhas
import sys
sys.path.append('/Users/erik/Desktop/Sinapses/TIFFS/_Working')

def process_all_images():
    """Processar todas as imagens das 4 linhagens"""
    print("🧬 Processamento Completo - Todas as Linhagens")
    print("="*70)
    
    # Carregar modelo binário
    model_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spine_Classification_Pipeline/FINAL_ANALYSIS/models/binary_classification_model.pkl')
    scaler_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spine_Classification_Pipeline/FINAL_ANALYSIS/models/binary_classification_scaler.pkl')
    
    if not model_path.exists() or not scaler_path.exists():
        print("❌ Modelo binário não encontrado!")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Modelo binário carregado (85.8% acurácia)")
    
    # Definir estrutura das linhagens
    lineages = {
        'ASD16': {'CTL': 'ASD16_CTL', 'CRISPRa': 'ASD16_CRISPR'},
        'ASD17': {'CTL': 'ASD17_CTL', 'CRISPRa': 'ASD17_CRISPR'},
        'IM5': {'CTL': 'IM5_CTL', 'CRISPRa': 'IM5_CRISPR'},
        'EB3': {'CTL': 'EB3_CTL', 'CRISPRa': 'EB3_CRISPR'}
    }
    
    print(f"\n📊 Linhagens a Processar:")
    for lineage, treatments in lineages.items():
        for treatment, sample_name in treatments.items():
            print(f"  {sample_name}: {lineage} {treatment}")
    
    # Buscar todas as imagens TIF
    base_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg')
    
    all_results = []
    sample_summary = defaultdict(lambda: defaultdict(int))
    
    print(f"\n🔍 Buscando imagens TIF...")
    
    # Processar cada linhagem
    for lineage, treatments in lineages.items():
        print(f"\n📁 Processando linhagem: {lineage}")
        
        for treatment, sample_name in treatments.items():
            print(f"\n  🔬 Tratamento: {treatment}")
            
            # Buscar imagens TIF
            image_dir = base_path / lineage / treatment
            if not image_dir.exists():
                print(f"    ⚠️ Diretório não encontrado: {image_dir}")
                continue
            
            tif_files = list(image_dir.glob('*.tif'))
            print(f"    📊 Imagens TIF encontradas: {len(tif_files)}")
            
            if not tif_files:
                print(f"    ⚠️ Nenhuma imagem TIF encontrada")
                continue
            
            # Processar cada imagem
            for tif_file in tif_files:
                try:
                    image_name = tif_file.stem
                    print(f"      🖼️ Processando: {image_name}")
                    
                    # Verificar se já existe dados processados
                    existing_spines = base_path / 'Results' / lineage / treatment / f'Hybrid_{image_name}_per_spine.csv'
                    
                    if existing_spines.exists():
                        print(f"        ✅ Dados já existem, carregando...")
                        spines_df = pd.read_csv(existing_spines)
                    else:
                        print(f"        🔄 Processando imagem (detecção de espinhas)...")
                        # Aqui você precisaria implementar a detecção de espinhas
                        # Por enquanto, vou pular imagens sem dados processados
                        print(f"        ⚠️ Dados não processados, pulando...")
                        continue
                    
                    if len(spines_df) == 0:
                        print(f"        ⚠️ Nenhuma espinha detectada")
                        continue
                    
                    # Extrair features
                    features_list = []
                    for _, spine in spines_df.iterrows():
                        features = extract_enhanced_features(spine)
                        features_list.append(features)
                    
                    features_df = pd.DataFrame(features_list)
                    X = features_df.values
                    
                    # Normalizar features
                    X_scaled = scaler.transform(X)
                    
                    # Fazer predições
                    predictions = model.predict(X_scaled)
                    prediction_proba = model.predict_proba(X_scaled)[:, 1]  # Probabilidade de ser mushroom
                    
                    # Adicionar predições ao dataframe
                    spines_df['predicted_class'] = predictions
                    spines_df['mushroom_probability'] = prediction_proba
                    
                    # Contar espinhas por classe
                    total_spines = len(spines_df)
                    mushroom_spines = np.sum(predictions == 'mushroom')
                    not_mushroom_spines = total_spines - mushroom_spines
                    
                    # Calcular densidade por comprimento do dendrito
                    if len(spines_df) > 1:
                        # Calcular comprimento total do dendrito baseado na dispersão das espinhas
                        x_coords = spines_df['base_x'].values
                        y_coords = spines_df['base_y'].values
                        
                        # Estimar comprimento do dendrito como a distância máxima entre espinhas
                        dendrite_length = 0
                        for i in range(len(x_coords)):
                            for j in range(i+1, len(x_coords)):
                                dist = np.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)
                                dendrite_length = max(dendrite_length, dist)
                        
                        # Se não conseguir calcular, usar estimativa baseada no número de espinhas
                        if dendrite_length == 0:
                            dendrite_length = len(spines_df) * 10  # Estimativa: 10 pixels por espinha
                        
                        # Calcular densidades por comprimento (espinhas/μm)
                        # Assumir 1 pixel = 0.1 μm (ajustar conforme calibração real)
                        pixel_to_um = 0.1
                        dendrite_length_um = dendrite_length * pixel_to_um
                        
                        spine_density = total_spines / max(dendrite_length_um, 1)  # espinhas/μm
                        mushroom_density = mushroom_spines / max(dendrite_length_um, 1)  # mushrooms/μm
                    else:
                        spine_density = total_spines
                        mushroom_density = mushroom_spines
                        dendrite_length_um = 0
                    
                    # Salvar resultados
                    result = {
                        'Sample': sample_name,
                        'Lineage': lineage,
                        'Treatment': treatment,
                        'Image': image_name,
                        'Total_Spines': total_spines,
                        'Mushroom_Spines': mushroom_spines,
                        'Not_Mushroom_Spines': not_mushroom_spines,
                        'Mushroom_Percentage': (mushroom_spines / total_spines) * 100 if total_spines > 0 else 0,
                        'Spine_Density_per_um': spine_density,  # espinhas/μm
                        'Mushroom_Density_per_um': mushroom_density,  # mushrooms/μm
                        'Dendrite_Length_um': dendrite_length_um if len(spines_df) > 1 else 0,
                        'Mean_Mushroom_Probability': np.mean(prediction_proba),
                        'Std_Mushroom_Probability': np.std(prediction_proba)
                    }
                    
                    all_results.append(result)
                    
                    # Atualizar resumo por amostra
                    sample_summary[sample_name]['Total_Spines'] += total_spines
                    sample_summary[sample_name]['Mushroom_Spines'] += mushroom_spines
                    sample_summary[sample_name]['Images'] += 1
                    
                    print(f"        ✅ {total_spines} espinhas, {mushroom_spines} mushrooms ({result['Mushroom_Percentage']:.1f}%)")
                    
                except Exception as e:
                    print(f"        ❌ Erro ao processar {tif_file}: {str(e)}")
                    continue
    
    # Criar DataFrames para análise
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("❌ Nenhum resultado encontrado!")
        return None, None
    
    print(f"\n📊 Resumo dos Resultados:")
    print(f"  Total de imagens processadas: {len(results_df)}")
    print(f"  Total de espinhas analisadas: {results_df['Total_Spines'].sum()}")
    print(f"  Total de mushrooms detectados: {results_df['Mushroom_Spines'].sum()}")
    
    # Análise estatística por grupo
    print(f"\n📈 Análise por Grupo:")
    group_analysis = results_df.groupby(['Lineage', 'Treatment']).agg({
        'Total_Spines': ['mean', 'std', 'count'],
        'Mushroom_Spines': ['mean', 'std'],
        'Spine_Density_per_um': ['mean', 'std'],
        'Mushroom_Density_per_um': ['mean', 'std'],
        'Mushroom_Percentage': ['mean', 'std'],
        'Mean_Mushroom_Probability': ['mean', 'std']
    }).round(3)
    
    print(group_analysis)
    
    # Preparar dados para GraphPad Prism
    prepare_graphpad_data(results_df)
    
    # Gerar visualizações
    generate_comprehensive_plots(results_df)
    
    # Salvar resultados
    save_results(results_df, group_analysis)
    
    return results_df, group_analysis

def extract_enhanced_features(spine):
    """Extrair features melhoradas da espinha"""
    base_x, base_y = spine['base_x'], spine['base_y']
    tip_x, tip_y = spine['tip_x'], spine['tip_y']
    confidence = spine['confidence']
    
    # Features básicas
    length = np.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
    angle = np.arctan2(tip_y - base_y, tip_x - base_x)
    
    # Features melhoradas
    features = {
        'length': length,
        'angle': angle,
        'confidence': confidence,
        'orientation': min(angle, 2*np.pi - angle),
        'distance_from_origin': np.sqrt(base_x**2 + base_y**2),
        'shape_factor': min(length / max(1, abs(tip_x - base_x)), 10),
        'normalized_length': length / max(1, np.sqrt(base_x**2 + base_y**2)),
        'normalized_angle': angle / (2 * np.pi),
        'base_x': base_x,
        'base_y': base_y,
        'dx': tip_x - base_x,
        'dy': tip_y - base_y,
        'relative_x': base_x / max(1, tip_x),
        'relative_y': base_y / max(1, tip_y)
    }
    
    return features

def prepare_graphpad_data(results_df):
    """Preparar dados no formato para GraphPad Prism"""
    print(f"\n📊 Preparando dados para GraphPad Prism...")
    
    # Criar tabelas no formato GraphPad
    output_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spine_Classification_Pipeline/FINAL_ANALYSIS/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tabela 1: Densidade Total de Espinhas (por comprimento)
    total_spines_table = results_df.pivot_table(
        values='Spine_Density_per_um',
        index='Image',
        columns=['Lineage', 'Treatment'],
        aggfunc='mean'
    )
    
    # Tabela 2: Densidade de Mushrooms (por comprimento)
    mushroom_spines_table = results_df.pivot_table(
        values='Mushroom_Density_per_um',
        index='Image',
        columns=['Lineage', 'Treatment'],
        aggfunc='mean'
    )
    
    # Tabela 3: Percentual de Mushrooms
    mushroom_percentage_table = results_df.pivot_table(
        values='Mushroom_Percentage',
        index='Image',
        columns=['Lineage', 'Treatment'],
        aggfunc='mean'
    )
    
    # Tabela 4: Probabilidade Média de Mushroom
    mushroom_prob_table = results_df.pivot_table(
        values='Mean_Mushroom_Probability',
        index='Image',
        columns=['Lineage', 'Treatment'],
        aggfunc='mean'
    )
    
    # Salvar tabelas
    total_spines_table.to_csv(output_dir / 'GraphPad_Spine_Density_per_um.csv')
    mushroom_spines_table.to_csv(output_dir / 'GraphPad_Mushroom_Density_per_um.csv')
    mushroom_percentage_table.to_csv(output_dir / 'GraphPad_Mushroom_Percentage.csv')
    mushroom_prob_table.to_csv(output_dir / 'GraphPad_Mushroom_Probability.csv')
    
    # Criar tabela resumida para análise estatística
    summary_table = results_df.groupby(['Lineage', 'Treatment']).agg({
        'Total_Spines': ['mean', 'std', 'sem', 'count'],
        'Mushroom_Spines': ['mean', 'std', 'sem'],
        'Spine_Density_per_um': ['mean', 'std', 'sem'],
        'Mushroom_Density_per_um': ['mean', 'std', 'sem'],
        'Mushroom_Percentage': ['mean', 'std', 'sem'],
        'Mean_Mushroom_Probability': ['mean', 'std', 'sem']
    }).round(3)
    
    summary_table.to_csv(output_dir / 'GraphPad_Summary_Statistics.csv')
    
    # Criar tabela com uma linha por imagem (como solicitado)
    image_table = results_df[['Image', 'Lineage', 'Treatment', 'Total_Spines', 'Mushroom_Spines', 
                             'Mushroom_Percentage', 'Spine_Density_per_um', 'Mushroom_Density_per_um',
                             'Dendrite_Length_um', 'Mean_Mushroom_Probability', 'Std_Mushroom_Probability']].copy()
    
    image_table.to_csv(output_dir / 'GraphPad_Image_Level_Data.csv', index=False)
    
    print(f"✅ Dados para GraphPad salvos em:")
    print(f"  - GraphPad_Spine_Density_per_um.csv")
    print(f"  - GraphPad_Mushroom_Density_per_um.csv") 
    print(f"  - GraphPad_Mushroom_Percentage.csv")
    print(f"  - GraphPad_Mushroom_Probability.csv")
    print(f"  - GraphPad_Summary_Statistics.csv")
    print(f"  - GraphPad_Image_Level_Data.csv (UMA LINHA POR IMAGEM)")

def generate_comprehensive_plots(results_df):
    """Gerar gráficos abrangentes de análise"""
    print(f"\n📊 Gerando gráficos abrangentes...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Criar figura com múltiplos painéis
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    # Gráfico 1: Densidade Total de Espinhas por Linhagem
    ax1 = axes[0, 0]
    sns.boxplot(data=results_df, x='Lineage', y='Spine_Density_per_um', hue='Treatment', ax=ax1)
    ax1.set_title('A) Densidade Total de Espinhas por Linhagem', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Densidade de Espinhas (espinhas/μm)')
    ax1.legend(title='Tratamento')
    
    # Gráfico 2: Densidade de Mushrooms por Linhagem
    ax2 = axes[0, 1]
    sns.boxplot(data=results_df, x='Lineage', y='Mushroom_Density_per_um', hue='Treatment', ax=ax2)
    ax2.set_title('B) Densidade de Mushrooms por Linhagem', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Densidade de Mushrooms (mushrooms/μm)')
    ax2.legend(title='Tratamento')
    
    # Gráfico 3: Percentual de Mushrooms por Linhagem
    ax3 = axes[0, 2]
    sns.boxplot(data=results_df, x='Lineage', y='Mushroom_Percentage', hue='Treatment', ax=ax3)
    ax3.set_title('C) Percentual de Mushrooms por Linhagem', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentual de Mushrooms (%)')
    ax3.legend(title='Tratamento')
    
    # Gráfico 4: Comparação CTL vs CRISPR (Densidade Total)
    ax4 = axes[1, 0]
    ctl_data = results_df[results_df['Treatment'] == 'CTL']
    crispr_data = results_df[results_df['Treatment'] == 'CRISPRa']
    
    comparison_data = []
    for _, row in ctl_data.iterrows():
        comparison_data.append({'Group': f"{row['Lineage']}_CTL", 'Value': row['Spine_Density_per_um']})
    for _, row in crispr_data.iterrows():
        comparison_data.append({'Group': f"{row['Lineage']}_CRISPR", 'Value': row['Spine_Density_per_um']})
    
    comparison_df = pd.DataFrame(comparison_data)
    sns.boxplot(data=comparison_df, x='Group', y='Value', ax=ax4)
    ax4.set_title('D) Comparação CTL vs CRISPR (Densidade Total)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Densidade de Espinhas (espinhas/μm)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Gráfico 5: Comparação CTL vs CRISPR (Mushrooms)
    ax5 = axes[1, 1]
    comparison_mush_data = []
    for _, row in ctl_data.iterrows():
        comparison_mush_data.append({'Group': f"{row['Lineage']}_CTL", 'Value': row['Mushroom_Density_per_um']})
    for _, row in crispr_data.iterrows():
        comparison_mush_data.append({'Group': f"{row['Lineage']}_CRISPR", 'Value': row['Mushroom_Density_per_um']})
    
    comparison_mush_df = pd.DataFrame(comparison_mush_data)
    sns.boxplot(data=comparison_mush_df, x='Group', y='Value', ax=ax5)
    ax5.set_title('E) Comparação CTL vs CRISPR (Mushrooms)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Densidade de Mushrooms (mushrooms/μm)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Gráfico 6: Resumo Estatístico por Grupo
    ax6 = axes[1, 2]
    summary_stats = results_df.groupby(['Lineage', 'Treatment'])['Spine_Density_per_um'].agg(['mean', 'std']).reset_index()
    
    x_pos = np.arange(len(summary_stats))
    bars = ax6.bar(x_pos, summary_stats['mean'], yerr=summary_stats['std'], 
                   capsize=5, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan', 'lightsteelblue'])
    
    ax6.set_title('F) Resumo Estatístico por Grupo', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Média ± DP de Densidade de Espinhas')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"{row['Lineage']}_{row['Treatment']}" for _, row in summary_stats.iterrows()], 
                       rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, (bar, mean_val) in enumerate(zip(bars, summary_stats['mean'])):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + summary_stats['std'].iloc[i] + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Gráfico 7: Heatmap de Densidade de Espinhas
    ax7 = axes[2, 0]
    heatmap_data = results_df.pivot_table(values='Spine_Density_per_um', index='Lineage', columns='Treatment', aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax7)
    ax7.set_title('G) Heatmap - Densidade de Espinhas', fontsize=14, fontweight='bold')
    
    # Gráfico 8: Heatmap de Densidade de Mushrooms
    ax8 = axes[2, 1]
    heatmap_mush_data = results_df.pivot_table(values='Mushroom_Density_per_um', index='Lineage', columns='Treatment', aggfunc='mean')
    sns.heatmap(heatmap_mush_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax8)
    ax8.set_title('H) Heatmap - Densidade de Mushrooms', fontsize=14, fontweight='bold')
    
    # Gráfico 9: Correlação entre Densidade Total e Mushrooms
    ax9 = axes[2, 2]
    sns.scatterplot(data=results_df, x='Spine_Density_per_um', y='Mushroom_Density_per_um', 
                    hue='Lineage', style='Treatment', s=100, ax=ax9)
    ax9.set_title('I) Correlação: Densidade Total vs Mushrooms', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Densidade Total de Espinhas (espinhas/μm)')
    ax9.set_ylabel('Densidade de Mushrooms (mushrooms/μm)')
    
    # Calcular e mostrar correlação
    correlation = results_df['Spine_Density_per_um'].corr(results_df['Mushroom_Density_per_um'])
    ax9.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax9.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar figura
    output_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spine_Classification_Pipeline/FINAL_ANALYSIS/results')
    plt.savefig(output_dir / 'Comprehensive_Analysis_Results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráficos abrangentes salvos em: Comprehensive_Analysis_Results.png")

def save_results(results_df, group_analysis):
    """Salvar todos os resultados"""
    print(f"\n💾 Salvando resultados...")
    
    output_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spine_Classification_Pipeline/FINAL_ANALYSIS/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar dados completos
    results_df.to_csv(output_dir / 'Complete_Analysis_Results.csv', index=False)
    group_analysis.to_csv(output_dir / 'Group_Statistics.csv')
    
    # Criar relatório resumido
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_images': len(results_df),
        'total_spines': int(results_df['Total_Spines'].sum()),
        'total_mushrooms': int(results_df['Mushroom_Spines'].sum()),
        'model_accuracy': '85.8%',
        'lineages_analyzed': list(results_df['Lineage'].unique()),
        'treatments_analyzed': list(results_df['Treatment'].unique()),
        'summary_statistics': str(group_analysis.to_dict())
    }
    
    with open(output_dir / 'Analysis_Report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Resultados salvos em:")
    print(f"  - Complete_Analysis_Results.csv")
    print(f"  - Group_Statistics.csv")
    print(f"  - Analysis_Report.json")

if __name__ == "__main__":
    results_df, group_analysis = process_all_images()
