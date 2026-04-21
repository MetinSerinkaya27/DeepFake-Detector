import cv2
import numpy as np
import os

def sahtecilik_yakala_web(input_path, output_path, algoritma_adi='SIFT'):
    resim = cv2.imread(input_path)
    if resim is None: return {"count": 0, "is_fake": False}
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # 1. Dedektör Seçimi
    if algoritma_adi == 'SIFT' or algoritma_adi == 'SURF':
        dedektor = cv2.SIFT_create()
    elif algoritma_adi == 'AKAZE':
        dedektor = cv2.AKAZE_create()
    elif algoritma_adi == 'ORB':
        dedektor = cv2.ORB_create(nfeatures=5000)
    else:
        dedektor = cv2.SIFT_create()

    kp, des = dedektor.detectAndCompute(gri, None)
    if des is None or len(des) < 3:
        cv2.imwrite(output_path, resim)
        return {"count": 0, "is_fake": False}

    # 2. Eşleştirme (KNN k=3)
    norm = cv2.NORM_HAMMING if algoritma_adi == 'ORB' else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des, des, k=3)

    sahte_img = resim.copy()
    count = 0
    for match in matches:
        if len(match) < 3: continue
        m_kendisi, m_klon, m_komsu = match
        
        # Ratio Test & Mesafe Filtresi
        if m_klon.distance < 0.75 * m_komsu.distance:
            pt1, pt2 = kp[m_klon.queryIdx].pt, kp[m_klon.trainIdx].pt
            dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            
            if dist > 40: 
                # Daha ince ve profesyonel çizgiler (thickness=1)
                cv2.line(sahte_img, tuple(map(int, pt1)), tuple(map(int, pt2)), (0, 0, 220), 1)
                count += 1

    # Resmi temiz şekilde kaydediyoruz (Yazısız)
    cv2.imwrite(output_path, sahte_img)
    return {"count": count, "is_fake": count > 10}