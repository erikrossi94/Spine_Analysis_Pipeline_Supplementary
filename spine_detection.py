#!/usr/bin/env python3
"""
Detecção e Classificação de Espinhas Dendríticas
===============================================

Módulo especializado para detecção automática e classificação morfológica
de espinhas dendríticas em imagens de microscopia confocal.

Baseado em métodos de processamento de imagem e análise morfológica.

Autor: Sistema de Análise Automatizada
Data: 2024
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, segmentation, filters
# from skimage.feature import peak_local_maxima  # Removido para compatibilidade
from skimage.morphology import disk, skeletonize
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SpineDetector:
    """Detector de espinhas dendríticas usando processamento de imagem"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pixel_size_um = config.get('pixel_size_um', 0.1)  # μm/pixel
        
    def detect_spines(self, image: np.ndarray, dendrite_mask: np.ndarray) -> List[Dict]:
        """
        Detectar espinhas em uma imagem usando processamento de imagem
        
        Args:
            image: Imagem de entrada (canal de espinhas)
            dendrite_mask: Máscara binária do dendrito
            
        Returns:
            Lista de dicionários com informações das espinhas detectadas
        """
        logger.info("Iniciando detecção de espinhas")
        
        # Pré-processamento da imagem
        processed_image = self._preprocess_image(image)
        
        # Detectar protrusões
        protrusions = self._detect_protrusions(processed_image, dendrite_mask)
        
        # Filtrar e refinar detecções
        filtered_spines = self._filter_detections(protrusions, dendrite_mask)
        
        # Medir propriedades morfológicas
        measured_spines = self._measure_spine_properties(filtered_spines, image)
        
        logger.info(f"Detectadas {len(measured_spines)} espinhas")
        return measured_spines
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Pré-processar imagem para detecção de espinhas"""
        
        # Garantir que o array seja C-contiguous
        image = np.ascontiguousarray(image)
        
        # Normalizar imagem
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Filtro gaussiano para suavizar
        sigma = self.config.get('gaussian_sigma', 1.0)
        image_smooth = ndimage.gaussian_filter(image_norm, sigma=sigma)
        
        # Filtro de realce de bordas (Laplacian of Gaussian)
        if self.config.get('use_log_filter', True):
            log_filtered = filters.laplace(image_smooth)
            image_enhanced = image_smooth + 0.3 * log_filtered
        else:
            image_enhanced = image_smooth
            
        return image_enhanced
        
    def _detect_protrusions(self, image: np.ndarray, dendrite_mask: np.ndarray) -> np.ndarray:
        """Detectar protrusões usando análise morfológica"""
        
        # Threshold adaptativo
        threshold = filters.threshold_otsu(image)
        binary_image = image > threshold
        
        # Operações morfológicas para limpar
        binary_image = morphology.binary_opening(binary_image, disk(1))
        binary_image = morphology.binary_closing(binary_image, disk(2))
        
        # Remover componentes conectados muito pequenos
        min_size = self.config.get('min_spine_size_pixels', 10)
        binary_image = morphology.remove_small_objects(binary_image, min_size=min_size)
        
        # Encontrar componentes conectados
        labeled_image = measure.label(binary_image)
        
        return labeled_image
        
    def _filter_detections(self, labeled_image: np.ndarray, dendrite_mask: np.ndarray) -> List[Dict]:
        """Filtrar detecções baseado em critérios morfológicos"""
        
        filtered_spines = []
        regions = measure.regionprops(labeled_image)
        
        for region in regions:
            # Verificar se está próximo ao dendrito
            if self._is_near_dendrite(region, dendrite_mask):
                
                # Calcular propriedades básicas
                spine_info = {
                    'label': region.label,
                    'area_pixels': region.area,
                    'area_um2': region.area * (self.pixel_size_um ** 2),
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'coords': region.coords,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity,
                    'extent': region.extent
                }
                
                # Filtrar por critérios básicos
                if self._passes_basic_filters(spine_info):
                    filtered_spines.append(spine_info)
                    
        return filtered_spines
        
    def _is_near_dendrite(self, region, dendrite_mask: np.ndarray, distance_threshold: int = 5) -> bool:
        """Verificar se a região está próxima ao dendrito"""
        
        # Dilatar máscara do dendrito
        dilated_dendrite = morphology.binary_dilation(dendrite_mask, disk(distance_threshold))
        
        # Verificar sobreposição
        region_mask = np.zeros_like(dendrite_mask, dtype=bool)
        region_mask[region.coords[:, 0], region.coords[:, 1]] = True
        
        overlap = np.logical_and(region_mask, dilated_dendrite).sum()
        return overlap > 0
        
    def _passes_basic_filters(self, spine_info: Dict) -> bool:
        """Aplicar filtros básicos de qualidade"""
        
        # Filtro de área mínima
        min_area_um2 = self.config.get('min_spine_area_um2', 0.1)
        if spine_info['area_um2'] < min_area_um2:
            return False
            
        # Filtro de solidez (forma compacta)
        min_solidity = self.config.get('min_solidity', 0.3)
        if spine_info['solidity'] < min_solidity:
            return False
            
        # Filtro de excentricidade (não muito alongado)
        max_eccentricity = self.config.get('max_eccentricity', 0.95)
        if spine_info['eccentricity'] > max_eccentricity:
            return False
            
        return True
        
    def _measure_spine_properties(self, spines: List[Dict], original_image: np.ndarray) -> List[Dict]:
        """Medir propriedades morfológicas detalhadas das espinhas"""
        
        measured_spines = []
        
        for spine in spines:
            # Extrair ROI da espinha
            bbox = spine['bbox']
            roi = original_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            spine_mask = np.zeros_like(original_image, dtype=bool)
            spine_mask[spine['coords'][:, 0], spine['coords'][:, 1]] = True
            spine_roi_mask = spine_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            # Medir propriedades adicionais
            spine_properties = self._analyze_spine_morphology(roi, spine_roi_mask, spine)
            measured_spines.append(spine_properties)
            
        return measured_spines
        
    def _analyze_spine_morphology(self, roi: np.ndarray, mask: np.ndarray, spine_info: Dict) -> Dict:
        """Analisar morfologia detalhada de uma espinha"""
        
        # Medir comprimento (distância máxima entre pontos)
        coords = np.where(mask)
        if len(coords[0]) < 2:
            return spine_info
            
        # Calcular distâncias entre todos os pares de pontos
        points = np.column_stack((coords[0], coords[1]))
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
                
        max_length_pixels = max(distances) if distances else 0
        max_length_um = max_length_pixels * self.pixel_size_um
        
        # Estimar diâmetro da cabeça (assumindo região mais brilhante)
        head_diameter_um = self._estimate_head_diameter(roi, mask)
        
        # Estimar largura do pescoço
        neck_width_um = self._estimate_neck_width(roi, mask)
        
        # Calcular fator de forma
        perimeter = self._calculate_perimeter(mask)
        shape_factor = (4 * np.pi * spine_info['area_pixels']) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Adicionar propriedades medidas
        spine_info.update({
            'length_um': max_length_um,
            'head_diameter_um': head_diameter_um,
            'neck_width_um': neck_width_um,
            'shape_factor': shape_factor,
            'perimeter_pixels': perimeter,
            'perimeter_um': perimeter * self.pixel_size_um
        })
        
        return spine_info
        
    def _estimate_head_diameter(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Estimar diâmetro da cabeça da espinha"""
        
        # Encontrar região mais brilhante (assumindo que é a cabeça)
        masked_roi = roi * mask
        max_intensity = np.max(masked_roi)
        
        # Threshold para região da cabeça
        head_threshold = max_intensity * 0.8
        head_mask = (masked_roi > head_threshold) & mask
        
        if np.sum(head_mask) == 0:
            return 0.0
            
        # Calcular diâmetro equivalente
        head_area = np.sum(head_mask)
        head_diameter_pixels = 2 * np.sqrt(head_area / np.pi)
        
        return head_diameter_pixels * self.pixel_size_um
        
    def _estimate_neck_width(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Estimar largura do pescoço da espinha"""
        
        # Skeletonizar para encontrar eixo principal
        skeleton = skeletonize(mask)
        
        if np.sum(skeleton) == 0:
            return 0.0
            
        # Encontrar pontos do skeleton
        skeleton_coords = np.where(skeleton)
        
        if len(skeleton_coords[0]) < 2:
            return 0.0
            
        # Calcular largura média perpendicular ao skeleton
        # (implementação simplificada)
        distances = []
        for i, (y, x) in enumerate(zip(skeleton_coords[0], skeleton_coords[1])):
            # Encontrar distância perpendicular ao skeleton neste ponto
            perp_distance = self._perpendicular_distance_to_boundary(mask, y, x)
            if perp_distance > 0:
                distances.append(perp_distance)
                
        neck_width_pixels = np.median(distances) if distances else 0
        return neck_width_pixels * self.pixel_size_um
        
    def _perpendicular_distance_to_boundary(self, mask: np.ndarray, y: int, x: int) -> float:
        """Calcular distância perpendicular do ponto ao limite da máscara"""
        
        # Implementação simplificada - encontrar distância ao limite
        # em direções perpendiculares ao skeleton
        
        h, w = mask.shape
        max_distance = min(h, w) // 4
        
        # Verificar em 4 direções principais
        distances = []
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            distance = 0
            ny, nx = y, x
            while (0 <= ny < h and 0 <= nx < w and 
                   mask[ny, nx] and distance < max_distance):
                ny += dy
                nx += dx
                distance += 1
                
            if distance > 0:
                distances.append(distance)
                
        return np.median(distances) if distances else 0
        
    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calcular perímetro da máscara"""
        
        # Usar transformada de distância para calcular perímetro
        from scipy.ndimage import distance_transform_edt
        
        # Inverter máscara para distância ao fundo
        inverted_mask = ~mask
        dist_transform = distance_transform_edt(inverted_mask)
        
        # Perímetro é aproximadamente o número de pixels com distância = 1
        perimeter = np.sum((dist_transform >= 0.5) & (dist_transform < 1.5))
        
        return perimeter

class SpineClassifier:
    """Classificador de espinhas baseado em critérios morfológicos"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.criteria = config.get('classification_criteria', {})
        
    def classify_spines(self, spines: List[Dict]) -> List[Dict]:
        """Classificar espinhas em tipos morfológicos"""
        
        classified_spines = []
        
        for spine in spines:
            spine_type = self._classify_single_spine(spine)
            spine['type'] = spine_type
            classified_spines.append(spine)
            
        return classified_spines
        
    def _classify_single_spine(self, spine: Dict) -> str:
        """Classificar uma única espinha"""
        
        # Extrair propriedades
        length = spine.get('length_um', 0)
        head_diameter = spine.get('head_diameter_um', 0)
        neck_width = spine.get('neck_width_um', 0)
        area = spine.get('area_um2', 0)
        
        # Critérios de classificação
        criteria = self.criteria
        
        # Mushroom: cabeça ≥0.6 μm e head/neck ≥1.5
        if (head_diameter >= criteria.get('mushroom', {}).get('head_diameter_min', 0.6) and
            head_diameter / max(neck_width, 0.1) >= criteria.get('mushroom', {}).get('head_neck_ratio_min', 1.5)):
            return 'mushroom'
            
        # Thin: comprimento ≥0.7 μm e cabeça <0.6 μm
        elif (length >= criteria.get('thin', {}).get('length_min', 0.7) and
              head_diameter < criteria.get('thin', {}).get('head_diameter_max', 0.6) and
              neck_width > 0):
            return 'thin'
            
        # Stubby: comprimento ≤0.5 μm, sem pescoço
        elif (length <= criteria.get('stubby', {}).get('length_max', 0.5) and
              neck_width == 0):
            return 'stubby'
            
        # Filopodia: comprimento ≥2.0 μm, sem cabeça
        elif (length >= criteria.get('filopodia', {}).get('length_min', 2.0) and
              head_diameter == 0):
            return 'filopodia'
            
        else:
            return 'unclassified'

class DendriteTracer:
    """Tracer de dendritos usando análise de imagem"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pixel_size_um = config.get('pixel_size_um', 0.1)
        
    def trace_dendrite(self, image: np.ndarray) -> Dict:
        """Traçar dendrito e calcular comprimento"""
        
        # Pré-processar imagem
        processed = self._preprocess_dendrite_image(image)
        
        # Skeletonizar
        skeleton = self._skeletonize_dendrite(processed)
        
        # Calcular comprimento
        length_um = self._calculate_skeleton_length(skeleton)
        
        # Extrair centerline
        centerline = self._extract_centerline(skeleton)
        
        return {
            'skeleton': skeleton,
            'centerline': centerline,
            'length_um': length_um,
            'length_pixels': np.sum(skeleton)
        }
        
    def _preprocess_dendrite_image(self, image: np.ndarray) -> np.ndarray:
        """Pré-processar imagem para traçado do dendrito"""
        
        # Garantir que o array seja C-contiguous
        image = np.ascontiguousarray(image)
        
        # Normalizar
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Filtro gaussiano
        sigma = self.config.get('dendrite_sigma', 2.0)
        image_smooth = ndimage.gaussian_filter(image_norm, sigma=sigma)
        
        # Threshold
        threshold = filters.threshold_otsu(image_smooth)
        binary = image_smooth > threshold
        
        # Operações morfológicas
        binary = morphology.binary_opening(binary, disk(2))
        binary = morphology.binary_closing(binary, disk(3))
        
        return binary
        
    def _skeletonize_dendrite(self, binary_image: np.ndarray) -> np.ndarray:
        """Skeletonizar dendrito"""
        
        # Remover componentes pequenos
        min_size = self.config.get('min_dendrite_size', 100)
        cleaned = morphology.remove_small_objects(binary_image, min_size=min_size)
        
        # Skeletonizar
        skeleton = skeletonize(cleaned)
        
        return skeleton
        
    def _calculate_skeleton_length(self, skeleton: np.ndarray) -> float:
        """Calcular comprimento do skeleton em μm"""
        
        # Contar pixels do skeleton
        skeleton_pixels = np.sum(skeleton)
        
        # Converter para μm
        length_um = skeleton_pixels * self.pixel_size_um
        
        return length_um
        
    def _extract_centerline(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Extrair centerline do skeleton"""
        
        # Encontrar coordenadas do skeleton
        coords = np.where(skeleton)
        centerline = list(zip(coords[0], coords[1]))
        
        return centerline

def create_default_config() -> Dict:
    """Criar configuração padrão para detecção de espinhas"""
    
    return {
        'pixel_size_um': 0.1,  # μm/pixel
        'gaussian_sigma': 1.0,
        'use_log_filter': True,
        'min_spine_size_pixels': 10,
        'min_spine_area_um2': 0.1,
        'min_solidity': 0.3,
        'max_eccentricity': 0.95,
        'dendrite_sigma': 2.0,
        'min_dendrite_size': 100,
        'classification_criteria': {
            'mushroom': {
                'head_diameter_min': 0.6,
                'head_neck_ratio_min': 1.5
            },
            'thin': {
                'head_diameter_max': 0.6,
                'length_min': 0.7
            },
            'stubby': {
                'length_max': 0.5
            },
            'filopodia': {
                'length_min': 2.0
            }
        }
    }
