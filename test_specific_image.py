#!/usr/bin/env python3
"""
특정 이미지의 라벨 좌표 변환 테스트
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from utils.dataloaders import LoadRGBTImagesAndLabels
from utils.general import check_img_size, colorstr
from utils.plots import plot_images

def visualize_labels_on_image(imgs, labels, save_path, title="Labels Visualization"):
    """이미지에 라벨을 그려서 시각화"""
    if isinstance(imgs, list):
        # RGBT 이미지의 경우 visible 이미지 사용
        img = imgs[1]  # visible 이미지
    else:
        img = imgs
    
    # Tensor인 경우 numpy로 변환
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # CHW -> HWC, BGR -> RGB, 0-1 -> 0-255
    if img.max() <= 1.0:
        img = (img.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
    else:
        img = img.transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기
    h, w = img.shape[:2]
    
    # 라벨 그리기
    img_with_labels = img.copy()
    
    if len(labels) > 0:
        for label in labels:
            cls, x_center, y_center, width, height = label[:5]
            
            # 정규화된 좌표를 픽셀 좌표로 변환
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            width_px = int(width * w)
            height_px = int(height * h)
            
            # 바운딩 박스 좌표
            x1 = x_center_px - width_px // 2
            y1 = y_center_px - height_px // 2
            x2 = x_center_px + width_px // 2
            y2 = y_center_px + height_px // 2
            
            # 경계 체크
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_with_labels, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 중심점 그리기
            cv2.circle(img_with_labels, (x_center_px, y_center_px), 3, (0, 255, 0), -1)
            
            # 클래스 정보 표시
            label_text = f'Class: {int(cls)}'
            cv2.putText(img_with_labels, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print(f"Label: class={cls}, center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
    
    # 저장
    # Calculate aspect ratio from actual image dimensions
    img_h, img_w = img_with_labels.shape[:2]
    aspect_ratio = img_w / img_h
    
    # Set figure size to maintain aspect ratio
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(img_with_labels)
    plt.title(f"{title}\nImage size: {img_w}x{img_h} (ratio: {aspect_ratio:.3f})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved label visualization: {save_path} (dimensions: {img_w}x{img_h})")

def test_specific_image():
    """특정 이미지로 라벨 좌표 변환 테스트"""
    
    # 설정 - KAIST 원본 비율 유지 (640x512)
    imgsz = 640  # width를 기준으로 설정
    target_image = "set00_V000_I02203"
    
    # 하이퍼파라미터 로드
    with open("data/hyps/hyp.scratch-low.yaml") as f:
        hyp = yaml.safe_load(f)
    
    print(f"Testing specific image: {target_image}")
    
    # 데이터셋 생성 (augmentation 없이)
    print("1. Loading dataset WITHOUT augmentation...")
    dataset_no_aug = LoadRGBTImagesAndLabels(
        path="datasets/kaist-rgbt/train-all-04.txt",
        img_size=imgsz,
        batch_size=1,
        augment=False,
        hyp=hyp,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="no_aug: "
    )
    
    # 데이터셋 생성 (augmentation 있이)
    print("2. Loading dataset WITH augmentation...")
    hyp_test = hyp.copy()
    hyp_test['fliplr'] = 1.0  # 100% 확률로 수평 뒤집기
    hyp_test['shear'] = 10    # 큰 전단 변환
    
    dataset_with_aug = LoadRGBTImagesAndLabels(
        path="datasets/kaist-rgbt/train-all-04.txt",
        img_size=imgsz,
        batch_size=1,
        augment=True,
        hyp=hyp_test,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="with_aug: "
    )
    
    # 특정 이미지 찾기
    target_index = None
    for i, im_file in enumerate(dataset_no_aug.im_files):
        if target_image in Path(im_file).stem:
            target_index = i
            print(f"Found target image at index {i}: {Path(im_file).name}")
            break
    
    if target_index is None:
        print(f"Could not find target image: {target_image}")
        return False
    
    # 원본 이미지와 라벨 가져오기
    print("\n3. Getting original image and labels...")
    imgs_no_aug, labels_no_aug, path_no_aug, shapes_no_aug, index_no_aug = dataset_no_aug[target_index]
    
    print(f"Original labels shape: {labels_no_aug.shape}")
    if labels_no_aug.shape[0] > 0:
        print("Original labels:")
        for i, label in enumerate(labels_no_aug):
            if label[0] == 0:  # 배치 인덱스가 0인 경우만
                cls, x_center, y_center, width, height = label[1:6].numpy()
                print(f"  Label {i}: class={cls:.0f}, center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
    
    # 증강된 이미지와 라벨 가져오기
    print("\n4. Getting augmented image and labels...")
    imgs_with_aug, labels_with_aug, path_with_aug, shapes_with_aug, index_with_aug = dataset_with_aug[target_index]
    
    print(f"Augmented labels shape: {labels_with_aug.shape}")
    if labels_with_aug.shape[0] > 0:
        print("Augmented labels:")
        for i, label in enumerate(labels_with_aug):
            if label[0] == 0:  # 배치 인덱스가 0인 경우만
                cls, x_center, y_center, width, height = label[1:6].numpy()
                print(f"  Label {i}: class={cls:.0f}, center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
    
    # 라벨을 numpy 배열로 변환 (시각화용)
    labels_no_aug_vis = []
    labels_with_aug_vis = []
    
    for label in labels_no_aug:
        if label[0] == 0:  # 배치 인덱스가 0인 경우만
            labels_no_aug_vis.append(label[1:6].numpy())
    
    for label in labels_with_aug:
        if label[0] == 0:  # 배치 인덱스가 0인 경우만
            labels_with_aug_vis.append(label[1:6].numpy())
    
    labels_no_aug_vis = np.array(labels_no_aug_vis) if labels_no_aug_vis else np.zeros((0, 5))
    labels_with_aug_vis = np.array(labels_with_aug_vis) if labels_with_aug_vis else np.zeros((0, 5))
    
    # 시각화
    print("\n5. Generating visualizations...")
    
    # 원본 이미지와 라벨
    visualize_labels_on_image(imgs_no_aug, labels_no_aug_vis, 
                             f"test_specific_{target_image}_original.jpg", 
                             f"Original Image: {target_image}")
    
    # 증강된 이미지와 라벨
    visualize_labels_on_image(imgs_with_aug, labels_with_aug_vis, 
                             f"test_specific_{target_image}_augmented.jpg", 
                             f"Augmented Image: {target_image}")
    
    print("\nVisualization files saved:")
    print(f"  - test_specific_{target_image}_original.jpg")
    print(f"  - test_specific_{target_image}_augmented.jpg")
    
    # 라벨 좌표 변화 분석
    if len(labels_no_aug_vis) > 0 and len(labels_with_aug_vis) > 0:
        print("\n6. Analyzing coordinate transformation:")
        orig_label = labels_no_aug_vis[0]
        aug_label = labels_with_aug_vis[0]
        
        print(f"Original center: ({orig_label[1]:.3f}, {orig_label[2]:.3f})")
        print(f"Augmented center: ({aug_label[1]:.3f}, {aug_label[2]:.3f})")
        print(f"X-coordinate change: {aug_label[1] - orig_label[1]:.3f}")
        print(f"Y-coordinate change: {aug_label[2] - orig_label[2]:.3f}")
        
        # 수평 뒤집기 확인 (x_center가 1.0 - original_x_center에 가까운지)
        expected_x_flip = 1.0 - orig_label[1]
        x_diff = abs(aug_label[1] - expected_x_flip)
        print(f"Expected X after flip: {expected_x_flip:.3f}")
        print(f"Actual X difference from expected: {x_diff:.3f}")
        
        if x_diff < 0.05:  # 5% 오차 범위
            print("✓ Horizontal flip transformation appears correct!")
        else:
            print("✗ Horizontal flip transformation may have issues.")
    
    return True

def compare_multiple_augmentation_types():
    """여러 증강 타입 비교"""
    
    # KAIST 원본 비율 유지
    imgsz = 640  # width를 기준으로 설정
    target_image = "set00_V000_I00983"
    
    with open("data/hyps/hyp.scratch-low.yaml") as f:
        hyp = yaml.safe_load(f)
    
    print(f"\nComparing multiple augmentation types for: {target_image}")
    
    # 다양한 증강 설정
    aug_configs = [
        {"name": "no_aug", "fliplr": 0.0, "shear": 0, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0},
        {"name": "flip_only", "fliplr": 1.0, "shear": 0, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0},
        {"name": "shear_only", "fliplr": 0.0, "shear": 15, "hsv_h": 0, "hsv_s": 0, "hsv_v": 0},
        {"name": "combined", "fliplr": 1.0, "shear": 10, "hsv_h": 0.05, "hsv_s": 0.8, "hsv_v": 0.6},
    ]
    
    for config in aug_configs:
        print(f"\nTesting: {config['name']}")
        
        # 설정 적용
        hyp_test = hyp.copy()
        hyp_test.update({k: v for k, v in config.items() if k != "name"})
        
        # 데이터셋 생성
        dataset = LoadRGBTImagesAndLabels(
            path="datasets/kaist-rgbt/train-all-04.txt",
            img_size=imgsz,
            batch_size=1,
            augment=(config["name"] != "no_aug"),
            hyp=hyp_test,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f"{config['name']}: "
        )
        
        # 특정 이미지 찾기
        target_index = None
        for i, im_file in enumerate(dataset.im_files):
            if target_image in Path(im_file).stem:
                target_index = i
                break
        
        if target_index is None:
            print(f"  Could not find target image")
            continue
        
        # 이미지와 라벨 가져오기
        imgs, labels, path, shapes, index = dataset[target_index]
        
        # 라벨을 numpy 배열로 변환
        labels_vis = []
        for label in labels:
            if label[0] == 0:  # 배치 인덱스가 0인 경우만
                labels_vis.append(label[1:6].numpy())
        
        labels_vis = np.array(labels_vis) if labels_vis else np.zeros((0, 5))
        
        # 시각화
        save_path = f"test_specific_{target_image}_{config['name']}.jpg"
        visualize_labels_on_image(imgs, labels_vis, save_path, 
                                 f"{config['name']}: {target_image}")
        
        print(f"  Saved: {save_path}")
        print(f"  Number of labels: {len(labels_vis)}")

if __name__ == "__main__":
    print("Testing specific image coordinate transformation...")
    
    # 특정 이미지 테스트
    test_specific_image()
    
    print("\n" + "="*50)
    
    # 다양한 증강 비교
    compare_multiple_augmentation_types()
    
    print("\nSpecific image test completed!")
