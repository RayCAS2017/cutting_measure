import cv2
import numpy as np
import gradio as gr  


def align(gray_ref, gray_reg):
    # 初始化ORB检测器 
    orb = cv2.ORB_create()  
    
    # 找到关键点和描述符  
    kp1, des1 = orb.detectAndCompute(gray_ref,None)  
    kp2, des2 = orb.detectAndCompute(gray_reg,None)  
    
    # 创建BFMatcher对象  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    
    # 匹配描述符  
    matches = bf.match(des1,des2)  
    
    # 按距离排序  
    matches = sorted(matches, key = lambda x:x.distance)  
    
    # 使用RANSAC算法找到单应性矩阵H  
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)  
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)  
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 

    return M 

def get_mask(gray, thr):

    gray = 1 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  
    # 二值化处理  
    _, thresholded = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY) 
    # 查找轮廓  
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    areas = [cv2.contourArea(contour) for contour in contours]  
    max_area_index = np.argmax(areas)  
    # 创建一个全黑的mask图像  
    mask = np.zeros_like(gray)    
    # 绘制轮廓到mask图像上  
    cv2.drawContours(mask, [contours[max_area_index]], -1, 255, thickness=cv2.FILLED) 

    return mask 


def blob_analysis(img, mask, x_size, y_size, fill_color=[255,0,0]):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask) 

    # 遍历每个blob，画出其轮廓并标注尺寸大小  
    min_area = 50
    for i in range(1, ret): 
       
        # 获取当前blob的尺寸和质心  
        x, y, w, h, area = stats[i]  
        if area > min_area:
            cx, cy = centroids[i]              
            # 画出blob的轮廓  
            color = (0, 255, 0)  # 红色  
            thickness = 4  
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)  
            
            # 标注尺寸大小  
            font = cv2.FONT_HERSHEY_SIMPLEX  
            # text = f"Size: {w*x_size}x{h*y_size}"  
            text = f"Size: {area*x_size*y_size}"  
            # text = f"Size: {w*h}"
            cv2.putText(img, text, (x-5, y-5), font, 4, (255, 0, 0), thickness)  

            img[labels==i] = fill_color

    return img
  
def cal_diff(rgb_ref, rgb_reg, text):

    print(text)
    settings = text.strip().split(",")
    thr = int(settings[0])
    x_size = float(settings[1])
    y_size = float(settings[2])
    gray_ref = cv2.cvtColor(rgb_ref, cv2.COLOR_BGR2GRAY)
    gray_reg = cv2.cvtColor(rgb_reg, cv2.COLOR_BGR2GRAY) 

    M = align(gray_reg, gray_ref)

    mask_ref = get_mask(gray_ref,thr)
    mask_reg = get_mask(gray_reg,thr)
    cv2.imwrite('./results/mask_ref.jpg', mask_ref)
    cv2.imwrite('./results/mask_reg.jpg', mask_reg)

    h,w = mask_ref.shape
    mask_reg2ref = cv2.warpPerspective(mask_reg, M, (w,h))
    cv2.imwrite('./results/mask_reg2ref.jpg', mask_reg2ref)
    h,w,_ = rgb_reg.shape
    rgb_reg2ref = cv2.warpPerspective(rgb_reg, M, (w,h))
    cv2.imwrite('./results/rgb_reg2ref.jpg', rgb_reg2ref)
    
    mask_diff = cv2.bitwise_xor(mask_ref, mask_reg2ref) 
    mask_diff_ref = cv2.bitwise_and(mask_diff, mask_ref) 
    mask_diff_reg = cv2.bitwise_and(mask_diff, mask_reg2ref)

    kernel = np.ones((5, 5), np.uint8)  
    # 进行腐蚀操作  
    mask_diff_ref = cv2.erode(mask_diff_ref, kernel, iterations=1)
    mask_diff_reg = cv2.erode(mask_diff_reg, kernel, iterations=1)  
    # 进行膨胀操作  
    mask_diff_ref = cv2.dilate(mask_diff_ref, kernel, iterations=1) 
    mask_diff_reg = cv2.dilate(mask_diff_reg, kernel, iterations=1)  
 

    cv2.imwrite('./results/mask_diff_ref.jpg', mask_diff_ref)
    cv2.imwrite('./results/mask_diff_reg.jpg', mask_diff_reg)


    output = blob_analysis(rgb_reg2ref, mask_diff_ref, x_size, y_size, fill_color=[255,0,0])
    output = blob_analysis(output, mask_diff_reg, x_size, y_size, fill_color=[0,0,255])

    cv2.imwrite('./results/output.jpg', output)

    return output

if __name__ == "__main__":
    # iface = gr.Interface(fn=cal_diff, inputs=["image", "image", 'text'],   
    #                  outputs=["image"], title="裁片测量")

    iface = gr.Interface(fn=cal_diff, 
                         inputs=[
                             gr.Image(label="模板图片"), 
                             gr.Image(label="待测图片"), 
                             gr.Textbox(label="设置：阈值，像素尺寸（x,y）", value="127,1,1")
                             ],   
                         outputs=[gr.Image(label="测量结果")], 
                         title="裁片测量"
                        )

    iface.launch()
    # rgb_ref = cv2.imread('ref.bmp')
    # rgb_reg = cv2.imread('reg.bmp')

    # output = cal_diff(rgb_ref, rgb_reg) 

