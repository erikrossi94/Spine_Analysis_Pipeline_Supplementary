#!/usr/bin/env python3
"""
Classificador Melhorado de Espinhas Dendríticas
Baseado em critérios morfológicos da literatura e classificações manuais
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ImprovedSpineClassifier:
    """Classificador de espinhas baseado em features morfológicas"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.class_names = ['mushroom', 'thin', 'stubby', 'filopodia', 'double_head', 'unclassified']
        
        # Critérios baseados na literatura (Pchitskaya & Bezprozvanny, 2020)
        self.literature_criteria = {
            'mushroom': {
                'min_head_diameter': 0.6,  # μm
                'min_head_neck_ratio': 1.5,
                'description': 'Cabeça grande e pescoço estreito'
            },
            'thin': {
                'min_length': 0.7,  # μm
                'max_head_diameter': 0.6,  # μm
                'description': 'Pescoço longo e cabeça pequena'
            },
            'stubby': {
                'max_length': 0.5,  # μm
                'description': 'Sem pescoço definido, cabeça pequena'
            },
            'filopodia': {
                'min_length': 2.0,  # μm
                'description': 'Extensão longa sem cabeça definida'
            },
            'double_head': {
                'description': 'Duas cabeças em um pescoço'
            }
        }
    
    def extract_morphological_features(self, spine_data):
        """Extrair features morfológicas das espinhas"""
        features = []
        
        for _, spine in spine_data.iterrows():
            # Calcular comprimento
            length_pixels = np.sqrt((spine['tip_x'] - spine['base_x'])**2 + 
                                  (spine['tip_y'] - spine['base_y'])**2)
            length_um = length_pixels * 0.1  # Assumindo 0.1 μm/pixel
            
            # Features básicas
            feature_vector = [
                length_um,  # Comprimento em μm
                length_pixels,  # Comprimento em pixels
                spine['confidence'],  # Confiança do detector
                # Adicionar mais features conforme necessário
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def classify_by_literature_criteria(self, spine_data):
        """Classificar espinhas usando critérios da literatura"""
        classifications = []
        
        for _, spine in spine_data.iterrows():
            # Calcular comprimento
            length_pixels = np.sqrt((spine['tip_x'] - spine['base_x'])**2 + 
                                  (spine['tip_y'] - spine['base_y'])**2)
            length_um = length_pixels * 0.1
            
            # Aplicar critérios da literatura
            if length_um >= 2.0:
                classification = 'filopodia'
            elif length_um <= 0.5:
                classification = 'stubby'
            elif length_um >= 0.7:
                # Para thin vs mushroom, precisamos de mais features
                # Por enquanto, usar heurística simples
                if spine['confidence'] > 0.7:
                    classification = 'mushroom'
                else:
                    classification = 'thin'
            else:
                classification = 'unclassified'
            
            classifications.append(classification)
        
        return classifications
    
    def train_with_manual_data(self, training_data):
        """Treinar modelo com dados manuais"""
        print("🧠 Treinando classificador com dados manuais...")
        
        # Extrair features
        X = self.extract_morphological_features(training_data)
        y = training_data['manual_classification'].values
        
        # Definir nomes das features
        self.feature_names = ['length_um', 'length_pixels', 'confidence']
        
        # Treinar modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Avaliar performance
        y_pred = self.model.predict(X)
        
        # Obter classes únicas nos dados
        unique_classes = np.unique(np.concatenate([y, y_pred]))
        
        print("\n📊 Performance do Modelo Treinado:")
        print(classification_report(y, y_pred, labels=unique_classes, target_names=unique_classes))
        
        # Matriz de confusão
        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_classes, yticklabels=unique_classes)
        plt.title('Matriz de Confusão - Classificador Treinado')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.tight_layout()
        
        # Salvar figura
        output_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/confusion_matrix_trained.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matriz de confusão salva em: {output_path}")
        
        return self.model
    
    def classify_spines(self, spine_data, method='hybrid'):
        """Classificar espinhas usando método especificado"""
        if method == 'literature':
            return self.classify_by_literature_criteria(spine_data)
        elif method == 'ml' and self.model is not None:
            X = self.extract_morphological_features(spine_data)
            return self.model.predict(X)
        elif method == 'hybrid':
            # Combinar literatura + ML
            lit_classifications = self.classify_by_literature_criteria(spine_data)
            
            if self.model is not None:
                X = self.extract_morphological_features(spine_data)
                ml_classifications = self.model.predict(X)
                
                # Combinar resultados (literatura tem prioridade para casos claros)
                hybrid_classifications = []
                for i, (lit, ml) in enumerate(zip(lit_classifications, ml_classifications)):
                    if lit in ['filopodia', 'stubby']:  # Critérios claros da literatura
                        hybrid_classifications.append(lit)
                    else:
                        hybrid_classifications.append(ml)
                
                return hybrid_classifications
            else:
                return lit_classifications
        else:
            raise ValueError(f"Método '{method}' não suportado")
    
    def evaluate_performance(self, test_data, method='hybrid'):
        """Avaliar performance do classificador"""
        predictions = self.classify_spines(test_data, method=method)
        actual = test_data['manual_classification'].values
        
        # Calcular métricas
        accuracy = np.mean(predictions == actual) * 100
        
        print(f"\n📈 Performance do Método '{method}':")
        print(f"Precisão Geral: {accuracy:.1f}%")
        
        # Relatório detalhado
        unique_classes = np.unique(np.concatenate([actual, predictions]))
        print("\nRelatório de Classificação:")
        print(classification_report(actual, predictions, labels=unique_classes, target_names=unique_classes))
        
        return accuracy, predictions

def load_training_data():
    """Carregar dados de treinamento"""
    training_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Results/manual_classifications_training.csv')
    
    if not training_path.exists():
        print("❌ Arquivo de treinamento não encontrado!")
        return None
    
    df = pd.read_csv(training_path)
    print(f"📥 Carregados {len(df)} exemplos de treinamento")
    
    return df

def load_all_spines_data():
    """Carregar todos os dados de espinhas"""
    results_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Results/ASD16/CRISPRa')
    
    all_spines = []
    for csv_file in results_dir.glob('*_per_spine.csv'):
        df = pd.read_csv(csv_file)
        all_spines.append(df)
    
    if not all_spines:
        print("❌ Nenhum arquivo de espinhas encontrado!")
        return None
    
    combined_df = pd.concat(all_spines, ignore_index=True)
    print(f"📥 Carregadas {len(combined_df)} espinhas totais")
    
    return combined_df

def main():
    """Função principal"""
    print("🔬 Classificador Melhorado de Espinhas Dendríticas")
    print("="*60)
    
    # Carregar dados
    training_data = load_training_data()
    if training_data is None:
        return
    
    all_spines = load_all_spines_data()
    if all_spines is None:
        return
    
    # Criar classificador
    classifier = ImprovedSpineClassifier()
    
    # Treinar com dados manuais
    classifier.train_with_manual_data(training_data)
    
    # Avaliar diferentes métodos
    print("\n" + "="*60)
    print("📊 AVALIAÇÃO DE DIFERENTES MÉTODOS")
    print("="*60)
    
    # Método 1: Apenas critérios da literatura
    print("\n1️⃣ Método: Critérios da Literatura")
    lit_accuracy, lit_predictions = classifier.evaluate_performance(training_data, method='literature')
    
    # Método 2: Machine Learning
    print("\n2️⃣ Método: Machine Learning")
    ml_accuracy, ml_predictions = classifier.evaluate_performance(training_data, method='ml')
    
    # Método 3: Híbrido
    print("\n3️⃣ Método: Híbrido (Literatura + ML)")
    hybrid_accuracy, hybrid_predictions = classifier.evaluate_performance(training_data, method='hybrid')
    
    # Resumo
    print("\n" + "="*60)
    print("📈 RESUMO DE PERFORMANCE")
    print("="*60)
    print(f"Critérios da Literatura: {lit_accuracy:.1f}%")
    print(f"Machine Learning:       {ml_accuracy:.1f}%")
    print(f"Método Híbrido:         {hybrid_accuracy:.1f}%")
    
    # Aplicar melhor método a todas as espinhas
    best_method = 'hybrid' if hybrid_accuracy >= max(lit_accuracy, ml_accuracy) else 'ml' if ml_accuracy >= lit_accuracy else 'literature'
    
    print(f"\n🎯 Melhor método: {best_method} ({max(lit_accuracy, ml_accuracy, hybrid_accuracy):.1f}%)")
    
    # Classificar todas as espinhas
    print(f"\n🔄 Aplicando melhor método a todas as {len(all_spines)} espinhas...")
    all_predictions = classifier.classify_spines(all_spines, method=best_method)
    
    # Atualizar dados
    all_spines['improved_type'] = all_predictions
    all_spines['classification_method'] = best_method
    
    # Salvar resultados
    output_path = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Results/improved_classifications.csv')
    all_spines.to_csv(output_path, index=False)
    
    print(f"✅ Resultados salvos em: {output_path}")
    
    # Estatísticas finais
    print(f"\n📊 Distribuição Final das Classificações:")
    final_counts = pd.Series(all_predictions).value_counts()
    for class_name, count in final_counts.items():
        percentage = count / len(all_predictions) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Classificação melhorada concluída!")

if __name__ == "__main__":
    main()
