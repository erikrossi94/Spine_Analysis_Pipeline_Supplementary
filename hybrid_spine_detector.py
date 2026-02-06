#!/usr/bin/env python3
"""
Detector Híbrido de Espinhas Dendríticas
========================================

Estratégia: Usar anotações Stage 1 (dendritos + espinhas) como guia para:
1. Segmentação de dendritos baseada em polilinhas anotadas
2. Detecção de espinhas por proximidade às polilinhas
3. Classificação ML com anotações de classe

Autor: Sistema Híbrido
Data: 2024
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports principais
import tifffile
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# scikit-image
from skimage import morphology, measure, filters, feature
from skimage.morphology import disk, skeletonize
from skimage.filters import gaussian
from skimage.draw import line as draw_line
from skimage.measure import regionprops

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSpineDetector:
    """Detector híbrido usando anotações Stage 1 como guia"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._create_default_config()
        self.results_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Results')
        self.data_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg')
        self.annotations_dir = Path('/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Annotations')
        
        # Criar estrutura de resultados
        self._setup_results_structure()
        
        logger.info("Detector Híbrido inicializado")
        
    def _create_default_config(self) -> Dict:
        """Configuração padrão para hIPSC"""
        return {
            'pixel_size_xy_nm': 50,
            'random_seed': 42,
            
            # Segmentação guiada
            'dendrite_dilation_px': 8,  # Dilatar polilinhas para criar máscara
            'spine_search_radius_px': 25,  # Raio de busca para espinhas (expandido para incluir soma)
            'min_spine_length_um': 0.3,
            'max_spine_length_um': 4.0,
            
            # Classificação ML
            'use_ml_classification': True,
            'ml_model': 'RandomForest',
            'ml_params': {'n_estimators': 200, 'max_depth': 10, 'random_state': 42},
            'cv_folds': 3,
            
            # Cores para overlay
            'overlay_colors': {
                'mushroom': 'red',
                'thin': 'blue', 
                'stubby': 'green',
                'filopodia': 'yellow',
                'unclassified': 'gray'
            }
        }
        
    def _setup_results_structure(self):
        """Configurar estrutura de diretórios"""
        for subdir in ['CSV', 'Overlays', 'Models']:
            (self.results_dir / subdir).mkdir(exist_ok=True)
            
    def run_hybrid_analysis(self, max_images: Optional[int] = None):
        """Executar análise híbrida completa"""
        logger.info("Iniciando análise híbrida de espinhas")
        
        # 1. Treinar classificador com anotações existentes
        classifier = self._train_classifier()
        
        # 2. Processar imagens com anotações Stage 1
        annotated_images = self._get_annotated_images()
        
        if max_images:
            annotated_images = annotated_images[:max_images]
            
        all_results = []
        
        for i, image_info in enumerate(annotated_images, 1):
            logger.info(f"Processando imagem {i}/{len(annotated_images)}: {image_info['basename']}")
            
            try:
                result = self._process_annotated_image(image_info, classifier)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar {image_info['basename']}: {str(e)}")
                
        # 3. Gerar resultados agregados
        self._generate_aggregated_results(all_results)
        
        logger.info(f"Análise híbrida concluída: {len(all_results)} imagens processadas")
        
    def _get_annotated_images(self) -> List[Dict]:
        """Obter imagens com anotações Stage 1"""
        images = []
        
        for spine_file in self.annotations_dir.glob('*_stage1_spines.csv'):
            basename = spine_file.name.replace('_stage1_spines.csv', '')
            
            # Verificar se existe arquivo de dendritos
            dendrite_file = self.annotations_dir / f'{basename}_stage1_dendrites.json'
            if not dendrite_file.exists():
                continue
                
            # Encontrar imagem correspondente
            for lineage in ['ASD16', 'ASD17', 'EB3', 'IM5']:
                for group in ['CRISPRa', 'CTL']:
                    image_path = self.data_dir / lineage / group / f'{basename}.tif'
                    if image_path.exists():
                        images.append({
                            'file_path': str(image_path),
                            'lineage': lineage,
                            'group': group,
                            'basename': basename,
                            'spine_annotations': str(spine_file),
                            'dendrite_annotations': str(dendrite_file)
                        })
                        break
                        
        logger.info(f"Encontradas {len(images)} imagens com anotações Stage 1")
        return images
        
    def _train_classifier(self) -> Optional[RandomForestClassifier]:
        """Treinar classificador com anotações existentes"""
        logger.info("Treinando classificador com anotações existentes")
        
        # Coletar dados de treinamento
        training_data = []
        training_labels = []
        
        for annotation_file in self.annotations_dir.glob('*_annotations.csv'):
            try:
                df = pd.read_csv(annotation_file)
                if 'class' in df.columns:
                    for _, row in df.iterrows():
                        # Extrair features morfológicas
                        features = self._extract_spine_features(row)
                        if features:
                            training_data.append(features)
                            training_labels.append(row['class'])
            except Exception as e:
                logger.warning(f"Erro ao processar {annotation_file}: {e}")
                
        if len(training_data) < 10:
            logger.warning("Poucos dados de treinamento - usando classificação por regras")
            return None
            
        # Treinar RandomForest
        X = np.array(training_data)
        y = np.array(training_labels)
        
        ml_params = self.config.get('ml_params', {'n_estimators': 200, 'max_depth': 10, 'random_state': 42})
        classifier = RandomForestClassifier(**ml_params)
        classifier.fit(X, y)
        
        # Validação cruzada
        cv_folds = self.config.get('cv_folds', 3)
        cv_scores = cross_val_score(classifier, X, y, cv=cv_folds, scoring='f1_macro')
        logger.info(f"F1-macro (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Salvar modelo
        model_path = self.results_dir / 'Models' / 'spine_classifier.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
            
        return classifier
        
    def _extract_spine_features(self, spine_data: pd.Series) -> Optional[List[float]]:
        """Extrair features morfológicas de uma espinha"""
        try:
            # Calcular comprimento
            length = np.sqrt((spine_data['tip_x'] - spine_data['base_x'])**2 + 
                           (spine_data['tip_y'] - spine_data['base_y'])**2)
            length_um = length * (self.config['pixel_size_xy_nm'] / 1000.0)
            
            # Calcular ângulo (orientação da espinha)
            angle = np.arctan2(spine_data['tip_y'] - spine_data['base_y'], 
                              spine_data['tip_x'] - spine_data['base_x'])
            angle_deg = np.degrees(angle) % 360
            
            # Features morfológicas robustas
            features = [
                length_um,                    # Comprimento
                angle_deg,                    # Ângulo de orientação
                spine_data.get('head_diameter_um', 0),  # Diâmetro da cabeça
                spine_data.get('neck_width_um', 0),     # Largura do pescoço
                spine_data.get('head_neck_ratio', 0),   # Razão cabeça/pescoço
                spine_data.get('shape_factor', 0),      # Fator de forma
                spine_data.get('area_um2', 0),          # Área
                spine_data.get('perimeter_um', 0)       # Perímetro
            ]
            
            return features
        except Exception:
            return None
            
    def _process_annotated_image(self, image_info: Dict, classifier: Optional[RandomForestClassifier]) -> Optional[Dict]:
        """Processar imagem usando anotações Stage 1"""
        
        # Carregar imagem
        image = tifffile.imread(image_info['file_path'])
        if len(image.shape) == 3:
            base_image = image[:, :, 1] if image.shape[2] >= 3 else image[:, :, 0]
        else:
            base_image = image
            
        # Carregar anotações
        dendrite_polylines = self._load_dendrite_annotations(image_info['dendrite_annotations'])
        spine_annotations = pd.read_csv(image_info['spine_annotations'])
        
        # Criar máscara de dendrito baseada nas polilinhas
        dendrite_mask = self._create_dendrite_mask_from_polylines(dendrite_polylines, base_image.shape)
        
        # Detectar espinhas por proximidade às polilinhas (versão anterior melhor)
        detected_spines = self._detect_spines_near_polylines(
            base_image, dendrite_polylines, spine_annotations
        )
        
        # Detectar espinhas na região do soma usando anotações
        soma_spines = self._detect_spines_in_annotated_soma(base_image, image_info)
        detected_spines.extend(soma_spines)
        
        # Classificar espinhas
        classified_spines = self._classify_detected_spines(detected_spines, classifier)
        
        # Calcular métricas
        density_metrics = self._calculate_density_metrics(classified_spines, dendrite_polylines)
        
        # Compilar resultados
        results = {
            'image_info': image_info,
            'spines': classified_spines,
            'density_metrics': density_metrics,
            'dendrite_mask': dendrite_mask,
            'original_image': base_image
        }
        
        # Exportar resultados
        self._export_results(results)
        
        return results
        
    def _load_dendrite_annotations(self, annotation_file: str) -> List[List[Tuple[float, float]]]:
        """Carregar polilinhas de dendrito"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        return data.get('dendrites', [])
        
    def _create_dendrite_mask_from_polylines(self, polylines: List[List[Tuple[float, float]]], image_shape: Tuple[int, int]) -> np.ndarray:
        """Criar máscara de dendrito a partir das polilinhas"""
        mask = np.zeros(image_shape, dtype=bool)
        
        for polyline in polylines:
            if len(polyline) < 2:
                continue
                
            # Desenhar polilinha
            for i in range(len(polyline) - 1):
                y0, x0 = int(round(polyline[i][0])), int(round(polyline[i][1]))
                y1, x1 = int(round(polyline[i+1][0])), int(round(polyline[i+1][1]))
                
                # Desenhar linha
                rr, cc = draw_line(
                    np.clip(y0, 0, image_shape[0]-1), 
                    np.clip(x0, 0, image_shape[1]-1),
                    np.clip(y1, 0, image_shape[0]-1), 
                    np.clip(x1, 0, image_shape[1]-1)
                )
                mask[rr, cc] = True
                
        # Dilatar para criar máscara mais robusta
        mask = morphology.binary_dilation(mask, disk(self.config['dendrite_dilation_px']))
        
        return mask
        
    def _detect_spines_near_polylines(self, image: np.ndarray, polylines: List[List[Tuple[float, float]]], 
                                     spine_annotations: pd.DataFrame) -> List[Dict]:
        """Detectar espinhas próximas às polilinhas (versão mais restritiva)"""
        detected_spines = []
        
        # Para cada espinha anotada, verificar se há sinal na imagem
        for _, spine in spine_annotations.iterrows():
            base_x, base_y = spine['base_x'], spine['base_y']
            tip_x, tip_y = spine['tip_x'], spine['tip_y']
            
            # Verificar se está próximo a alguma polilinha (mais restritivo)
            if self._is_near_polyline((base_y, base_x), polylines):
                # Validação adicional: verificar se a espinha tem características válidas
                if self._validate_spine_annotation((base_y, base_x), (tip_y, tip_x), image):
                    # Extrair região da espinha
                    spine_region = self._extract_spine_region(image, (base_y, base_x), (tip_y, tip_x))
                    
                    if spine_region is not None:
                        detected_spines.append({
                            'id': len(detected_spines),
                            'base_point': (base_y, base_x),
                            'tip_point': (tip_y, tip_x),
                            'region': spine_region,
                            'annotated': True
                        })
                    
        return detected_spines
        
    def _validate_spine_annotation(self, base: Tuple[float, float], tip: Tuple[float, float], image: np.ndarray) -> bool:
        """Validar se anotação de espinha é válida"""
        try:
            # Verificar comprimento
            length = np.sqrt((tip[0] - base[0])**2 + (tip[1] - base[1])**2)
            length_um = length * (self.config['pixel_size_xy_nm'] / 1000.0)
            
            if length_um < self.config['min_spine_length_um'] or length_um > self.config['max_spine_length_um']:
                return False
                
            # Verificar se há sinal na região
            y1, x1 = int(base[0]), int(base[1])
            y2, x2 = int(tip[0]), int(tip[1])
            
            if 0 <= y1 < image.shape[0] and 0 <= x1 < image.shape[1] and \
               0 <= y2 < image.shape[0] and 0 <= x2 < image.shape[1]:
                
                # Verificar intensidade na base e ponta
                base_intensity = image[y1, x1]
                tip_intensity = image[y2, x2]
                
                # Deve ter sinal razoável
                if base_intensity < np.percentile(image, 30) or tip_intensity < np.percentile(image, 30):
                    return False
                    
            return True
        except Exception:
            return False
        
    def _detect_spines_in_annotated_soma(self, image: np.ndarray, image_info: Dict) -> List[Dict]:
        """Detectar espinhas na região do soma usando anotações manuais"""
        soma_spines = []
        
        # Carregar anotações do soma
        soma_annotation_path = self._get_soma_annotation_path(image_info)
        if not soma_annotation_path or not os.path.exists(soma_annotation_path):
            return soma_spines
            
        try:
            with open(soma_annotation_path, 'r') as f:
                soma_data = json.load(f)
                
            soma_regions = soma_data.get('soma_regions', [])
            if not soma_regions:
                return soma_spines
                
            # Para cada região do soma, detectar espinhas
            for region in soma_regions:
                x, y, width, height = region['x'], region['y'], region['width'], region['height']
                
                # Converter para coordenadas de pixel
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + width), int(y + height)
                
                # Garantir que está dentro da imagem
                x1 = max(0, min(x1, image.shape[1]))
                y1 = max(0, min(y1, image.shape[0]))
                x2 = max(0, min(x2, image.shape[1]))
                y2 = max(0, min(y2, image.shape[0]))
                
                # Extrair região do soma
                soma_region = image[y1:y2, x1:x2]
                
                # Detectar candidatos a espinhas na região do soma
                candidates = self._find_spine_candidates_in_soma_region(soma_region, (x1, y1))
                
                for candidate in candidates:
                    if self._validate_spine_candidate(candidate, image):
                        soma_spines.append({
                            'id': len(soma_spines) + 2000,  # IDs diferentes para espinhas do soma
                            'base_point': candidate['base'],
                            'tip_point': candidate['tip'],
                            'region': candidate.get('region'),
                            'annotated': False,
                            'source': 'soma_auto_detection'
                        })
                        
        except Exception as e:
            print(f"Erro ao processar anotações do soma: {e}")
            
        return soma_spines
        
    def _get_soma_annotation_path(self, image_info: Dict) -> Optional[str]:
        """Obter caminho para anotações do soma"""
        try:
            basename = Path(image_info['image_path']).stem
            soma_dir = Path("/Users/erik/Desktop/Sinapses/TIFFS/_Working/Spines_Reorg/Annotations/Soma")
            soma_file = soma_dir / f"{basename}_soma.json"
            return str(soma_file) if soma_file.exists() else None
        except Exception:
            return None
            
    def _find_spine_candidates_in_soma_region(self, soma_region: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """Encontrar candidatos a espinhas na região do soma"""
        candidates = []
        
        if soma_region.size == 0:
            return candidates
            
        # Encontrar bordas da região do soma
        edges = self._find_edges(soma_region)
        
        # Encontrar pontos de alta intensidade nas bordas
        edge_coords = np.column_stack(np.where(edges))
        
        for coord in edge_coords[::3]:  # Amostrar para velocidade
            y, x = coord[0], coord[1]
            
            # Converter para coordenadas globais
            global_y = y + offset[1]
            global_x = x + offset[0]
            
            # Tentar encontrar ponta da espinha
            tip = self._find_spine_tip_from_base(soma_region, (y, x), offset)
            if tip:
                candidates.append({
                    'base': (global_y, global_x),
                    'tip': tip,
                    'region': self._extract_spine_region_from_soma(soma_region, (y, x), tip, offset)
                })
                
        return candidates
        
    def _find_edges(self, region: np.ndarray) -> np.ndarray:
        """Encontrar bordas da região"""
        from scipy import ndimage
        
        # Usar filtro de Sobel para detectar bordas
        sobel_x = ndimage.sobel(region, axis=1)
        sobel_y = ndimage.sobel(region, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Threshold para bordas
        threshold = np.percentile(edges, 70)
        return edges > threshold
        
    def _find_spine_tip_from_base(self, region: np.ndarray, base: Tuple[int, int], offset: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Encontrar ponta da espinha a partir da base na região do soma"""
        try:
            y, x = base
            h, w = region.shape
            
            # Buscar em direções radiais
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for radius in range(3, 20):
                    tip_y = int(y + radius * np.sin(angle))
                    tip_x = int(x + radius * np.cos(angle))
                    
                    if 0 <= tip_y < h and 0 <= tip_x < w:
                        # Verificar se há sinal na ponta
                        if region[tip_y, tip_x] > np.percentile(region, 60):
                            # Converter para coordenadas globais
                            global_tip_y = tip_y + offset[1]
                            global_tip_x = tip_x + offset[0]
                            return (global_tip_y, global_tip_x)
            return None
        except Exception:
            return None
            
    def _extract_spine_region_from_soma(self, soma_region: np.ndarray, base: Tuple[int, int], 
                                      tip: Tuple[int, int], offset: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extrair região da espinha a partir da região do soma"""
        try:
            # Converter coordenadas globais para locais
            local_base = (base[0], base[1])
            local_tip = (tip[0] - offset[1], tip[1] - offset[0])
            
            # Criar região retangular
            y1 = min(local_base[0], local_tip[0])
            y2 = max(local_base[0], local_tip[0])
            x1 = min(local_base[1], local_tip[1])
            x2 = max(local_base[1], local_tip[1])
            
            # Adicionar margem
            margin = 2
            y1 = max(0, y1 - margin)
            y2 = min(soma_region.shape[0], y2 + margin)
            x1 = max(0, x1 - margin)
            x2 = min(soma_region.shape[1], x2 + margin)
            
            return soma_region[y1:y2, x1:x2]
        except Exception:
            return None
        
    def _detect_spines_in_soma_region(self, image: np.ndarray, polylines: List[List[Tuple[float, float]]]) -> List[Dict]:
        """Detectar espinhas automaticamente na região do soma"""
        soma_spines = []
        
        # Encontrar região do soma (área de alta intensidade)
        soma_mask = self._find_soma_region(image)
        
        if not soma_mask.any():
            return soma_spines
            
        # Encontrar candidatos a espinhas na região do soma
        candidates = self._find_spine_candidates_in_soma(image, soma_mask, polylines)
        
        for candidate in candidates:
            # Validar candidato
            if self._validate_spine_candidate(candidate, image):
                soma_spines.append({
                    'id': len(soma_spines) + 1000,  # IDs diferentes para espinhas do soma
                    'base_point': candidate['base'],
                    'tip_point': candidate['tip'],
                    'region': candidate.get('region'),
                    'annotated': False,  # Detectada automaticamente
                    'source': 'soma_detection'
                })
                
        return soma_spines
        
    def _find_soma_region(self, image: np.ndarray) -> np.ndarray:
        """Encontrar região do soma baseada em intensidade"""
        # Threshold para áreas de alta intensidade
        threshold = np.percentile(image, 75)
        soma_mask = image > threshold
        
        # Limpeza morfológica
        soma_mask = morphology.remove_small_objects(soma_mask, min_size=500)
        soma_mask = morphology.binary_closing(soma_mask, disk(5))
        
        return soma_mask
        
    def _find_spine_candidates_in_soma(self, image: np.ndarray, soma_mask: np.ndarray, 
                                     polylines: List[List[Tuple[float, float]]]) -> List[Dict]:
        """Encontrar candidatos a espinhas na região do soma"""
        candidates = []
        
        # Encontrar bordas do soma
        soma_edges = morphology.binary_dilation(soma_mask, disk(2)) & ~soma_mask
        
        # Encontrar pontos de alta intensidade nas bordas
        edge_coords = np.column_stack(np.where(soma_edges))
        
        for coord in edge_coords[::5]:  # Amostrar para velocidade
            y, x = coord[0], coord[1]
            
            # Verificar se está próximo a alguma polilinha
            if self._is_near_polyline((y, x), polylines):
                # Tentar encontrar ponta da espinha
                tip = self._find_spine_tip_from_base(image, (y, x))
                if tip:
                    candidates.append({
                        'base': (y, x),
                        'tip': tip,
                        'region': self._extract_spine_region(image, (y, x), tip)
                    })
                    
        return candidates
        
    def _find_spine_tip_from_base(self, image: np.ndarray, base: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Encontrar ponta da espinha a partir da base"""
        try:
            y, x = base
            h, w = image.shape
            
            # Buscar em direções radiais
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                for radius in range(5, 30):
                    tip_y = int(y + radius * np.sin(angle))
                    tip_x = int(x + radius * np.cos(angle))
                    
                    if 0 <= tip_y < h and 0 <= tip_x < w:
                        # Verificar se há sinal na ponta
                        if image[tip_y, tip_x] > np.percentile(image, 60):
                            return (tip_y, tip_x)
            return None
        except Exception:
            return None
            
    def _validate_spine_candidate(self, candidate: Dict, image: np.ndarray) -> bool:
        """Validar se candidato é uma espinha real"""
        try:
            base = candidate['base']
            tip = candidate['tip']
            
            # Verificar comprimento
            length = np.sqrt((tip[0] - base[0])**2 + (tip[1] - base[1])**2)
            length_um = length * (self.config['pixel_size_xy_nm'] / 1000.0)
            
            if length_um < self.config['min_spine_length_um'] or length_um > self.config['max_spine_length_um']:
                return False
                
            # Verificar intensidade
            if candidate.get('region') is not None:
                region = candidate['region']
                if region.mean() < np.percentile(image, 40):
                    return False
                    
            return True
        except Exception:
            return False
        
    def _is_near_polyline(self, point: Tuple[float, float], polylines: List[List[Tuple[float, float]]]) -> bool:
        """Verificar se ponto está próximo a alguma polilinha"""
        search_radius = self.config['spine_search_radius_px']
        
        for polyline in polylines:
            for poly_point in polyline:
                distance = np.sqrt((point[0] - poly_point[0])**2 + (point[1] - poly_point[1])**2)
                if distance <= search_radius:
                    return True
        return False
        
    def _is_in_soma_region(self, point: Tuple[float, float], image: np.ndarray) -> bool:
        """Detectar se ponto está na região do soma (área de alta intensidade)"""
        try:
            y, x = int(round(point[0])), int(round(point[1]))
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                # Verificar intensidade local (soma geralmente tem sinal mais forte)
                local_patch = image[max(0, y-10):min(image.shape[0], y+11), 
                                  max(0, x-10):min(image.shape[1], x+11)]
                if local_patch.size > 0:
                    local_intensity = local_patch.mean()
                    # Se intensidade local está no top 30%, provavelmente é soma
                    return local_intensity > np.percentile(image, 70)
            return False
        except Exception:
            return False
        
    def _extract_spine_region(self, image: np.ndarray, base: Tuple[float, float], tip: Tuple[float, float]) -> Optional[np.ndarray]:
        """Extrair região da espinha na imagem"""
        try:
            # Calcular bounding box
            y_min = max(0, int(min(base[0], tip[0]) - 10))
            y_max = min(image.shape[0], int(max(base[0], tip[0]) + 10))
            x_min = max(0, int(min(base[1], tip[1]) - 10))
            x_max = min(image.shape[1], int(max(base[1], tip[1]) + 10))
            
            region = image[y_min:y_max, x_min:x_max]
            
            # Verificar se há sinal suficiente
            if region.mean() > np.percentile(image, 30):
                return region
            return None
        except Exception:
            return None
            
    def _classify_detected_spines(self, spines: List[Dict], classifier: Optional[RandomForestClassifier]) -> List[Dict]:
        """Classificar espinhas detectadas"""
        classified_spines = []
        
        for spine in spines:
            # Extrair features
            features = self._extract_spine_features_from_region(spine)
            
            if features and classifier:
                # Classificação ML
                prediction = classifier.predict([features])[0]
                probability = classifier.predict_proba([features]).max()
            else:
                # Classificação por regras (fallback)
                prediction = self._classify_by_rules(spine)
                probability = 0.5
                
            spine['type'] = prediction
            spine['confidence'] = probability
            classified_spines.append(spine)
            
        return classified_spines
        
    def _extract_spine_features_from_region(self, spine: Dict) -> Optional[List[float]]:
        """Extrair features de uma região de espinha"""
        try:
            region = spine.get('region')
            if region is None:
                return None
                
            # Calcular features morfológicas básicas
            base = spine.get('base_point', (0, 0))
            tip = spine.get('tip_point', (0, 0))
            
            # Comprimento
            length = np.sqrt((tip[0] - base[0])**2 + (tip[1] - base[1])**2)
            length_um = length * (self.config['pixel_size_xy_nm'] / 1000.0)
            
            # Ângulo
            angle = np.arctan2(tip[0] - base[0], tip[1] - base[1])
            angle_deg = np.degrees(angle) % 360
            
            # Features da região (expandir para 8 features)
            features = [
                length_um,                    # Comprimento
                angle_deg,                    # Ângulo
                region.mean(),                # Intensidade média
                region.std(),                 # Desvio padrão
                region.max(),                 # Intensidade máxima
                region.min(),                 # Intensidade mínima
                np.percentile(region, 75) - np.percentile(region, 25),  # IQR
                region.size                   # Tamanho da região
            ]
            
            return features
        except Exception:
            return None
            
    def _classify_by_rules(self, spine: Dict) -> str:
        """Classificação por regras (fallback)"""
        # Implementar regras básicas baseadas na literatura
        return 'unclassified'
        
    def _calculate_density_metrics(self, spines: List[Dict], polylines: List[List[Tuple[float, float]]]) -> Dict:
        """Calcular métricas de densidade"""
        # Calcular comprimento total das polilinhas
        total_length_px = 0
        for polyline in polylines:
            for i in range(len(polyline) - 1):
                p1, p2 = polyline[i], polyline[i+1]
                total_length_px += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
        total_length_um = total_length_px * (self.config['pixel_size_xy_nm'] / 1000.0)
        
        # Densidade por 10 μm
        density_per_10um = (len(spines) / total_length_um) * 10.0 if total_length_um > 0 else 0
        
        # Contar por tipo
        type_counts = {}
        for spine in spines:
            spine_type = spine.get('type', 'unclassified')
            type_counts[spine_type] = type_counts.get(spine_type, 0) + 1
            
        return {
            'total_spines': len(spines),
            'dendrite_length_um': total_length_um,
            'density_per_10um': density_per_10um,
            'type_counts': type_counts
        }
        
    def _export_results(self, results: Dict):
        """Exportar resultados"""
        image_info = results['image_info']
        spines = results['spines']
        density_metrics = results['density_metrics']
        
        # Criar diretório específico
        lineage_dir = self.results_dir / image_info['lineage'] / image_info['group']
        lineage_dir.mkdir(parents=True, exist_ok=True)
        
        basename = image_info['basename']
        
        # CSV resumo
        summary_data = {
            'lineage': image_info['lineage'],
            'group': image_info['group'],
            'basename': basename,
            'dendrite_length_um': density_metrics['dendrite_length_um'],
            'n_spines_total': density_metrics['total_spines'],
            'density_per_10um': density_metrics['density_per_10um'],
            **{f'{k}_count': v for k, v in density_metrics['type_counts'].items()}
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_path = lineage_dir / f'Hybrid_{basename}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # CSV detalhado por espinha
        if spines:
            spines_data = []
            for spine in spines:
                spine_data = {
                    'spine_id': spine.get('id', ''),
                    'type': spine.get('type', ''),
                    'confidence': spine.get('confidence', 0),
                    'base_x': spine.get('base_point', (np.nan, np.nan))[1],
                    'base_y': spine.get('base_point', (np.nan, np.nan))[0],
                    'tip_x': spine.get('tip_point', (np.nan, np.nan))[1],
                    'tip_y': spine.get('tip_point', (np.nan, np.nan))[0],
                    'lineage': image_info['lineage'],
                    'group': image_info['group'],
                    'image_basename': basename
                }
                spines_data.append(spine_data)
                
            spines_df = pd.DataFrame(spines_data)
            spines_path = lineage_dir / f'Hybrid_{basename}_per_spine.csv'
            spines_df.to_csv(spines_path, index=False)
            
        # Criar overlay visual
        self._create_hybrid_overlay(results, lineage_dir, basename)
        
        logger.info(f"Resultados híbridos exportados para {lineage_dir}")
        
    def _create_hybrid_overlay(self, results: Dict, output_dir: Path, basename: str):
        """Criar overlay visual híbrido"""
        try:
            import matplotlib.pyplot as plt
            
            image_info = results['image_info']
            original_image = results['original_image']
            spines = results['spines']
            dendrite_mask = results['dendrite_mask']
            
            # Normalizar imagem
            if original_image.max() > original_image.min():
                base_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            else:
                base_image = original_image
                
            # Criar figura
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(base_image, cmap='gray', alpha=0.9)
            
            # Desenhar máscara de dendrito
            ax.imshow(dendrite_mask, cmap='Blues', alpha=0.3)
            
            # Cores para cada tipo
            colors = self.config.get('overlay_colors', {
                'mushroom': 'red',
                'thin': 'blue', 
                'stubby': 'green',
                'filopodia': 'yellow',
                'unclassified': 'gray'
            })
            
            # Plotar espinhas
            for i, spine in enumerate(spines):
                spine_type = spine.get('type', 'unclassified')
                color = colors.get(spine_type, 'gray')
                
                # Diferentes estilos para espinhas anotadas vs detectadas
                is_annotated = spine.get('annotated', True)
                line_style = '-' if is_annotated else '--'
                line_width = 3 if is_annotated else 2
                alpha = 0.9 if is_annotated else 0.7
                
                base = spine.get('base_point')
                tip = spine.get('tip_point')
                
                if base and tip:
                    # Linha base→ponta
                    ax.plot([base[1], tip[1]], [base[0], tip[0]], 
                           color=color, linewidth=line_width, alpha=alpha, linestyle=line_style)
                    
                    # Marcadores
                    marker_size = 60 if is_annotated else 40
                    ax.scatter(base[1], base[0], c='white', s=marker_size, marker='s', 
                              alpha=alpha, edgecolors=color, linewidth=2)
                    ax.scatter(tip[1], tip[0], c='white', s=100, marker='^', 
                              alpha=alpha, edgecolors=color, linewidth=2)
                    
                    # ID da espinha e classificação
                    center_y, center_x = (base[0] + tip[0]) / 2, (base[1] + tip[1]) / 2
                    ax.text(center_x, center_y, f"{i}", fontsize=10, color='white', 
                           weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=color, alpha=0.8))
                    
                    # Classificação na ponta
                    ax.text(tip[1] + 8, tip[0] - 8, spine_type[0].upper(), 
                           fontsize=12, color=color, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            
            # Configurar plot
            ax.set_title(f'Detecção Híbrida: {basename}\n'
                        f'Lineage: {image_info["lineage"]}, Group: {image_info["group"]}\n'
                        f'Total Spines: {len(spines)}', fontsize=14)
            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Y (pixels)', fontsize=12)
            
            # Informações de densidade
            density_info = results['density_metrics']
            info_text = (f'Dendrite Length: {density_info["dendrite_length_um"]:.2f} μm\n'
                        f'Density: {density_info["density_per_10um"]:.2f} spines/10μm\n'
                        f'Types: {density_info["type_counts"]}')
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                   facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Salvar overlay
            overlay_dir = output_dir / 'overlays'
            overlay_dir.mkdir(exist_ok=True)
            overlay_path = overlay_dir / f'hybrid_{basename}.png'
            plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Overlay híbrido salvo: {overlay_path}")
            
        except Exception as e:
            logger.error(f"Erro ao criar overlay híbrido: {e}")
            
    def _generate_aggregated_results(self, all_results: List[Dict]):
        """Gerar resultados agregados"""
        if not all_results:
            return
            
        # Compilar dados
        all_summaries = []
        all_spines = []
        
        for result in all_results:
            image_info = result['image_info']
            density_metrics = result['density_metrics']
            spines = result['spines']
            
            # Resumo da imagem
            summary_data = {
                'lineage': image_info['lineage'],
                'group': image_info['group'],
                'basename': image_info['basename'],
                'dendrite_length_um': density_metrics['dendrite_length_um'],
                'n_spines_total': density_metrics['total_spines'],
                'density_per_10um': density_metrics['density_per_10um'],
                **{f'{k}_count': v for k, v in density_metrics['type_counts'].items()}
            }
            all_summaries.append(summary_data)
            
            # Espinhas individuais
            for spine in spines:
                spine_data = {
                    'lineage': image_info['lineage'],
                    'group': image_info['group'],
                    'image_basename': image_info['basename'],
                    'spine_id': spine.get('id', ''),
                    'type': spine.get('type', ''),
                    'confidence': spine.get('confidence', 0),
                    'base_x': spine.get('base_point', (np.nan, np.nan))[1],
                    'base_y': spine.get('base_point', (np.nan, np.nan))[0],
                    'tip_x': spine.get('tip_point', (np.nan, np.nan))[1],
                    'tip_y': spine.get('tip_point', (np.nan, np.nan))[0]
                }
                all_spines.append(spine_data)
                
        # Salvar agregados
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = self.results_dir / 'CSV' / 'hybrid_summary_by_image.csv'
            summary_df.to_csv(summary_path, index=False)
            
        if all_spines:
            spines_df = pd.DataFrame(all_spines)
            spines_path = self.results_dir / 'CSV' / 'hybrid_all_spines_combined.csv'
            spines_df.to_csv(spines_path, index=False)
            
        logger.info("Resultados agregados híbridos gerados")

def main():
    """Função principal"""
    config = {
        'pixel_size_xy_nm': 50,
        'random_seed': 42,
        'use_ml_classification': True,
        'dendrite_dilation_px': 8,
        'spine_search_radius_px': 15,
        'min_spine_length_um': 0.3,
        'max_spine_length_um': 4.0
    }
    
    detector = HybridSpineDetector(config)
    detector.run_hybrid_analysis(max_images=3)  # Testar com 3 imagens primeiro
    
    logger.info("Análise híbrida executada com sucesso!")

if __name__ == "__main__":
    main()
