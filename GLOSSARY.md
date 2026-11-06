# GLOSSARY —— 中英术语对照表

> 全课程译名统一以此表为准。术语首次出现时，画面与字幕写「中文（English）」。

| 英文 | 中文 | 首现集 | 说明 |
|---|---|---|---|
| computer vision | 计算机视觉 | E009 |  |
| image formation | 图像形成 | E013 |  |
| additive / subtractive colors | 加色 / 减色 | E020 | 光加色、颜料减色 |
| tri-stimulus values | 三刺激值 | E020 | 三原色理论的量的表述 |
| cones | 视锥细胞 | E020 | 三种视锥=三个数 |
| metamers | 同色异谱 | E020 | 光谱不同、三刺激值相同 |
| standard observer | 标准观察者 | E020 | CIE 匹配实验产物 |
| chromaticity coordinates | 色度坐标 | E020 | x+y+z=1 |
| spectral locus | 光谱轨迹 | E020 | 色度图马蹄形边界 |
| purple line | 紫线 | E020 | 轨迹两端连线，非光谱色 |
| gamut | 色域 | E020 | 三原色可再现范围 |
| white point | 白点 | E020 | E=等能白 |
| demosaicing | 去马赛克 | E020 | Bayer 插值重建全彩 |
| CFA (color filter array) | 彩色滤镜阵列 | E020 | 即 Bayer 阵列 |
| companding | 压扩 | E020 | 传递前压缩、恢复后扩张 |
| chroma subsampling | 色度子采样 | E020 | 4:2:0 等，人眼色觉带宽低 |
| point operators | 点算子 | E021 | 逐像素套同一函数 g=h(f) |
| neighborhood operators | 邻域算子 | E021 | 看邻居，§3.2 起展开 |
| point processes | 点过程 | E021 | 此指点运算，非概率论同名概念 |
| gain | 增益 | E021 | 式 3.3 的 a，斜率 |
| bias | 偏置 | E021 | 式 3.3 的 b，平移 |
| 1×1 convolution | 一乘一卷积 | E021 | 点算子在 CNN 的名字（原书「有时被叫作」，p.111 脚注） |
| grayscale | 灰度 | E021 |  |
| contrast | 对比度 | E021 | 明暗差距 |
| brightness | 亮度 | E021 | 整体明暗 |
| graded density filter | 渐变灰滤镜 | E021 | 原书例子：空间可变增益 |
| superposition principle | 叠加原理 | E021 | 线性判据 |
| band-pass | 带通 | E021 | 混合算子的频域视角；E023 起指向「band-pass filter 带通滤波」 |
| cross-dissolve | 交叉溶解 | E021 | 转场 |
| linear blend | 线性混合 | E021 | (1−α)f₀+αf₁ |
| gamma correction | 伽马校正 | E021 | γ≈2.2，式 3.7 |
| hue | 色相 | E021 | 颜色的「种类」 |
| saturation | 饱和度 | E021 | 颜色的「浓淡」 |
| chromaticity | 色度 | E021 | 泛指；坐标定义见 E020 |
| color ratios | 颜色比率 | E021 | 通道比值守恒 |
| color balancing | 颜色平衡 | E021 |  |
| color twist | 色彩扭转 | E021 | 3×3 线性颜色变换 |
| matting | 抠图 | E021 | 求 α |
| compositing | 合成 | E021 | 用 α 贴进另一幅图 |
| alpha channel | α 通道 | E021 |  |
| opacity | 不透明度 | E021 | α 的语义 |
| jaggies | 锯齿 | E021 | 二值不透明度的边缘瑕疵 |
| feathering | 羽化 | E021 | 讲者用语（原书未用）：边界平滑过渡 |
| over operator | over 算子 | E021 | C=(1−α)B+αF，式 3.8 |
| foreground / background | 前景 / 背景 | E021 | over 的 F 与 B |
| premultiplied | 预乘 | E021 | 存 αF，合成链更稳 |
| RGBA | RGBA | E021 | 三色+α 四通道 |
| local color consistency | 局部颜色一致性 | E021 | 旋转带 α 图像的要求 |
| glass reflection | 玻璃反射 | E021 | 直接相加的合成，Figure 3.6 |
| transparent motion | 透明运动 | E021 |  |
| blue-screen matting | 蓝屏抠图 | E021 | 传统抠图，§10.4 综述 |
| difference matting | 差分抠图 | E021 | 已知背景差分求 α |
| histogram equalization | 直方图均衡 | E021 | E022 展开，本集预告首现 |
| middle gray | 中间灰 | E022 | 可显示范围的中心灰（两条启发式之一） |
| cumulative distribution function (CDF) | 累积分布函数 | E022 | 式 3.9 的 c(I)，逐级累加除以 N |
| percentile | 百分位 | E022 | 原书考试成绩比喻（p.117） |
| lookup table | 查找表 | E022 | 逐级查 c(I) 定映射 |
| noise amplification | 噪声放大 | E022 | 暗区噪声被增亮放大（p.117） |
| partial equalization | 部分均衡 | E022 | f=αc+(1−α)I 线性混合（p.117） |
| blocking artifacts | 块效应 | E022 | 块边界亮度跳变（p.117） |
| moving window | 滑动窗口 | E022 | 逐像素重算 M×M 直方图，太慢 |
| raster scan | 光栅扫描 | E022 | 逐行遍历，只更新进出块像素 |
| adaptive histogram equalization (AHE) | 自适应直方图均衡 | E022 | 块间平滑插值（Pizer et al. 1987） |
| CLAHE | 限制对比度自适应直方图均衡 | E022 | contrast-limited/gain-limited 同义；已进 OpenCV（p.118 脚注） |
| bilinear blending | 双线性混合 | E022 | 式 3.10 四查找表按 (s,t) 加权 |
| corner-based variant | 角点变体 | E022 | 查找表放块角（Figure 3.9b，Ex 3.8） |
| soft histogramming | 软直方图 | E022 | 式 3.11 计数按权重拆给四顶点；SIFT/词汇树同款 |
| tonal adjustment | 色调调整 | E022 | §3.1.5 应用节（p.119） |
| solarization | 曝光反转 | E022 | 原书括注 quantization 量化（Ex 3.6，p.181） |
| linear filtering | 线性滤波 | E022 | §3.2 标题预告（p.119），E023 展开 |
| histogram | 直方图 | E022 | 各灰度级像素计数（p.116 脚注 3） |
| dynamic range | 动态范围 | E022 | 可显示暗亮跨度（p.116 病例设问） |
| full equalization | 全均衡 | E022 | f(I)=c(I)（p.117） |
| muddy | 发灰 | E022 | 原书 muddy looking（p.117） |
| local equalization | 局部均衡 | E022 | 分块各调各的（p.117），原书随后引出块间插值 AHE（p.118） |
| P² (2D projective space) | 二维射影空间 | E013 | 增广向量齐次化 |
| vanishing point | 消失点 | E013 | 平行线交于无穷远点成像 |
| line at infinity | 无穷远线 | E013 |  |
| Plücker coordinates | 普吕克坐标 | E013 | 直线表示 |
| conic | 二次曲线 | E013 | 别名圆锥曲线 |
| quadric surface | 二次曲面 | E013 |  |
| Hough transform | 霍夫变换 | E013 | 参数空间投票 |
| projective depth | 射影深度 | E013 | 齐次第四分量 |
| skew-symmetric matrix | 斜对称矩阵 | E013 | 即叉积矩阵 [t]× |
| special orthogonal group SO(n) | 特殊正交群 | E014 | SO(2)/SO(3)：旋转群 |
| special Euclidean group SE(n) | 特殊欧氏群 | E014 | SE(2)/SE(3)：刚体运动群 |
| scaled rotation | 缩放旋转 | E014 | 相似变换的分解 |
| nested groups / subgroup | 嵌套群/子群 | E014 | Table 2.1 族谱读法 |
| co-vector | 余向量 | E014 | 射影变换下行作用对象 |
| adjoint | 伴随 | E014 | 群作用换基 |
| Euler angles | 欧拉角 | E015 | 万向锁缺陷 |
| gimbal lock | 万向锁 | E015 | 欧拉角退化 |
| axis/angle representation | 轴角表示 | E015 | ω=θn̂ |
| Rodrigues formula | 罗德里格斯公式 | E015 | 轴角↔旋转矩阵 |
| exponential twist | 指数扭转 | E015 | se(3)→SE(3) 矩阵指数 |
| Lie algebra se(3) | 李代数 | E015 | 切空间 |
| unit quaternion | 单位四元数 | E015 | 双重覆盖（对径点同旋转） |
| slerp | 球面线性插值 | E015 | 四元数插值 |
| orthography | 正交投影 | E015 | 弱透视 |
| scaled orthography | 缩放正交 | E015 |  |
| para-perspective | 平行透视 | E015 | 每帧仿射近似 |
| telecentric lens | 远心镜头 | E015 | 正交投影实镜 |
| normalized device coordinates | 归一化设备坐标 | E015 | 透视除法后 |
| z-buffer | 深度缓冲 | E015 | 可见性 |
| inverse depth | 逆深度 | E015 | 与视差等价、无穷远稳定 |
| pinhole model | 针孔模型 | E015 |  |
| calibration matrix | 标定矩阵 | E016 | K：内参 |
| intrinsics / extrinsics | 内参 / 外参 | E016 | K 与 (R,t) |
| principal point | 主点 | E016 | K 中 (c_x,c_y) |
| optical center / nodal point | 光心 / 节点 | E016 |  |
| skew (parameter) | 倾斜参数 | E016 | K 非矩形像素项 |
| field of view | 视场角 | E016 | 与焦距换算 |
| plane plus parallax | 平面加视差 | E016 | 单应+残余视差分解 |
| object-centered projection | 物体中心投影 | E016 | (s,ηz) 参数化 |
| foreshortening | 透视缩短（前缩） | E016 | E018 作前缩因子 cosθ |
| radial distortion | 径向畸变 | E017 | κ₁κ₂ 参数 |
| barrel / pincushion distortion | 桶形 / 枕形畸变 | E017 | κ 正负 |
| tangential distortion | 切向畸变 | E017 | 透镜偏心 |
| Brown model | 布朗模型 | E017 | 径向+切向合 model |
| fisheye lens | 鱼眼镜头 | E017 |  |
| equi-distance projection | 等距投影 | E017 | f-theta 镜头 |
| anamorphic lens | 变形镜头 | E017 | 非方形像素来源 |
| undistort | 去畸变 | E017 | 查表重采样 |
| chromatic aberration | 色差 | E017 | 横/纵两型，E019 展开 |
| BRDF (bidirectional reflectance distribution function) | 双向反射分布函数 | E018 | 互易性、各向同性 |
| reflectance equation | 反射方程 | E018 | 渲染方程原型 |
| irradiance | 辐照度 | E018 | 到达表面 |
| radiance | 辐亮度 | E018 | 沿方向离开 |
| Lambertian (diffuse) | 朗伯（漫反射） | E018 | 与视角无关 |
| specular reflection | 镜面反射 | E018 |  |
| micro-facet | 微面元模型 | E018 | 高光统计模型 |
| Phong shading | Phong 着色 | E018 | ambient+diffuse+specular |
| dichromatic reflection model | 双色反射模型 | E018 | 体反射+界面反射 |
| ray tracing | 光线追踪 | E018 |  |
| radiosity | 辐射度 | E018 | 形状因子、全局多次反射 |
| environment map | 环境贴图 | E018 | 远场光照 |
| attenuation | 消光 | E018 | 介质吸收散射 |
| albedo | 反照率 | E018 | 漫反射系数 |
| thin lens | 薄透镜 | E019 |  |
| lens law | 透镜定律 | E019 | 1/zi+1/zo=1/f |
| circle of confusion | 弥散圆 | E019 | 焦外光斑 |
| depth of field | 景深 | E019 |  |
| f-number (f-stop) | f 值 | E019 | N=f/D |
| stop (photography) | 挡 | E019 | 面积减半 |
| exposure value (EV) | 曝光值 | E019 |  |
| Seidel aberrations | 赛德尔像差 | E019 | 五种三阶像差 |
| transverse / longitudinal chromatic aberration | 横向 / 纵向色差 | E019 | 缩放差/焦点差 |
| catadioptric | 折反射系统 | E019 | 透镜+反射镜 |
| vignetting | 渐晕/暗角 | E019 | 自然（cos⁴）/机械；E142 补标定方法与校正 |
| fundamental radiometric relation | 基本辐射度关系 | E019 | 像面辐照度 E=Lπ/4·(d/zi)²·cos⁴ |
| magnification | 放大率 | E019 |  |
| diopter | 屈光度 | E019 | 1/f（米⁻¹） |
| MTF (modulation transfer function) | 调制传递函数 | E019 | 第一零点估焦 |
| image processing | 图像处理 | E021 |  |
| pixel | 像素 | E009 |  |
| vector | 向量 | E001 |  |
| matrix | 矩阵 | E001 |  |
| linear algebra | 线性代数 | E001 |  |
| dot product | 点积 | E001 |  |
| identity matrix | 单位矩阵 | E001 |  |
| inverse matrix | 逆矩阵 | E001 |  |
| norm | 范数 | E001 |  |
| bilinear interpolation | 双线性插值 | E001 | 实验代码使用，E030 展开 |
| homogeneous coordinates | 齐次坐标 | E001（预告）/E002（正式定义） |  |
| augmented vector | 增广向量 | E002 | (x, y, 1) 形式 |
| ideal point / point at infinity | 理想点 / 无穷远点 | E002 | 两个说法同义 |
| cross product | 叉积 | E002 |  |
| rigid (Euclidean) transformation | 刚体变换（欧氏变换） | E002 | 表2.1 族谱 |
| similarity transform | 相似变换 | E002 |  |
| affine transformation | 仿射变换 | E002 |  |
| projective transformation | 射影变换 | E002 | 别名：透视变换/单应（E074） |
| degrees of freedom | 自由度 | E002 |  |
| derivative | 导数 | E003 |  |
| partial derivative | 偏导数 | E003 |  |
| gradient | 梯度 | E003 |  |
| gradient descent | 梯度下降 | E003（直觉）/E041（训练循环） |  |
| learning rate | 学习率 | E003 | 原书 A.3 线搜索里叫步长 α |
| Jacobian (matrix) | 雅可比矩阵 | E003（预告）/E039（展开） | J = ∂f/∂p |
| residual | 残差 | E003 | 原书式(A.24) |
| condition number | 条件数 | E003（碗有多扁）/E004（病态问题） | 原书脚注9：最大/最小特征值之比 |
| normal equations | 正规方程 | E003（提及）/E004（展开） | 原书式(A.30)：(AᵀA)x = Aᵀb |
| Cholesky factorization | 乔列斯基分解 | E004 | 原书式(A.31)：C = AᵀA = RᵀR |
| QR factorization | QR 分解 | E004 | 原书式(A.34)：更稳但慢一个常数因子 |
| inverse problem | 逆问题 | E009 |  |
| optical illusion | 视错觉 | E009 |  |
| Necker cube | 内克尔立方 | E009 |  |
| Müller-Lyer illusion | 缪勒-莱尔错觉 | E009 |  |
| Hermann grid | 赫尔曼网格 | E009 |  |
| OCR (optical character recognition) | 光学字符识别 | E009 |  |
| Bayesian | 贝叶斯 | E009 |  |
| SVD (singular value decomposition) | 奇异值分解 | E001 | 附录 A.1.1 |
| eigenvalue | 特征值 | E001 |  |
| least squares | 最小二乘 | E004/E034 |  |
| RANSAC | 随机抽样一致 | E036 | 读英文音 |
| optimization | 优化 | E031 |  |
| loss function | 损失函数 | E041 |  |
| backpropagation | 反向传播 | E042 |  |
| overfitting | 过拟合 | E043 |  |
| CNN (convolutional neural network) | 卷积神经网络 | E044 |  |
| Transformer | Transformer | E047 | 不翻译 |
| ViT (vision transformer) | 视觉 Transformer | E048 |  |
| recognition | 识别 | E053 |  |
| object detection | 目标检测 | E056 |  |
| semantic segmentation | 语义分割 | E058 |  |
| feature | 特征 | E063 |  |
| keypoint / interest point | 特征点 | E063 |  |
| descriptor | 描述子 | E022（terms 提及）/E066（展开） |  |
| SIFT | 尺度不变特征变换 | E022（terms 提及）/E065（展开） | 读英文音 |
| homography | 单应 | E074 | 单应矩阵/单应变换 |
| panorama stitching | 全景拼接 | E075 |  |
| bundle adjustment | 捆扎调整 | E077 | 别名：捆绑调整、光束法平差（E011 历史集用「光束法平差」） |
| optical flow | 光流 | E080 |  |
| Lucas-Kanade | Lucas-Kanade（卢卡斯-卡纳德） | E073 | 简称 LK |
| HDR (high dynamic range) | 高动态范围 | E091 |  |
| tone mapping | 色调映射 | E092 |  |
| super-resolution | 超分辨率 | E093 |  |
| computational photography | 计算摄影 | E089 |  |
| structure from motion (SfM) | 运动恢复结构 | E099 |  |
| SLAM | 同步定位与建图 | E106 | 读英文音 |
| epipolar geometry | 对极几何 | E100 |  |
| fundamental matrix | 基础矩阵 | E100 |  |
| essential matrix | 本质矩阵 | E100 |  |
| triangulation | 三角化 | E102 |  |
| stereo matching | 立体匹配 | E110 |  |
| disparity | 视差 | E015（正式）/E109（立体匹配展开） |  |
| depth map | 深度图 | E109 |  |
| monocular depth estimation | 单目深度估计 | E113 |  |
| point cloud | 点云 | E050 | 首现更正 E118→E050（E050 §5.5.1 点云谱系口播+卡行实际首现 |
| Poisson reconstruction | 泊松重建 | E118 |  |
| ICP (iterative closest point) | 迭代最近点 | E120 |  |
| NeRF (neural radiance field) | 神经辐射场 | E121 | 读英文音 |
| 3D Gaussian Splatting | 三维高斯泼溅 | E123 | Splatting 不译 |
| view interpolation | 视图插值 | E127 |  |
| light field | 光场 | E013（E013 terms 提及）/E128（展开） | 首现集修正 2026-10-28（原 E128） |
| kernel | 核 | E023 | 权重表 h(k,l)，别名掩模（p.120） |
| mask | 掩模 | E023 | 核的别名（p.120） |
| filter coefficients | 滤波系数 | E023 | 核内权重（p.120） |
| correlation | 相关 | E023 | f⊗h 核不翻转直接滑（式 3.13，p.120） |
| convolution | 卷积 | E023 | f∗h 偏移翻转=核转 180°（式 3.14/3.15，p.120） |
| impulse response | 冲激响应 | E023 | h∗δ=h；δ 相关给镜像（p.120，Try this yourself 为原书）；E024 展开 IIR/FIR（p.130） |
| impulse signal | 冲激信号 | E023 | 全零图上原点一点亮（p.120） |
| linear shift-invariant (LSI) | 线性移不变 | E023 | 叠加+平移不变，behaves the same everywhere（式 3.16/3.17，p.122） |
| shift-variant kernel | 移变核 | E023 | h(k,l;i,j)；景深散焦例（式 3.18，p.122） |
| sparse matrix | 稀疏矩阵 | E023 | 大片元素为零；卷积⇔g=Hf（式 3.19，p.122） |
| boundary effects | 边界效应 | E023 | 邻域越界，valid 输出缩一圈（p.123）；LeNet 反采纳（脚注 6） |
| zero padding | 补零 | E023 | 图外置 0，模糊后边缘变暗（p.123）；公式留 Ex 3.9 |
| constant padding | 补常数 | E023 | 图外填边框色（p.123） |
| clamp (replicate) padding | 复制边缘 | E023 | 重复边缘像素，边值向内传播（p.123） |
| (cyclic) wrap padding | 环形绕回 | E023 | 环面配置；图形学叫 wrapping mode/texture addressing mode（pp.123–124） |
| mirror padding | 镜像翻转 | E023 | 沿图像边缘反射（p.123）；是否重复边缘像素原书未规定，本集实验取不重复口径（讲者注） |
| extension padding | 外插延伸 | E023 | 2f(0)−f(k) 外推，边界像素保持不动（p.124） |
| normalized zero padding | 归一化补零 | E023 | 补零后连 alpha 模糊再相除去暗角（pp.123–124） |
| soft alpha | 软 alpha | E023 | alpha-matted 抠图的透明度通道（p.123 图注） |
| box filter | 方块核 | E023 | K×K 平均=全一核再缩放（p.125） |
| running sum | 运行和 | E023 | 滑窗加最新减最旧（pp.125–126） |
| summed area table | 求和面积表 | E023 | 积分图；E024 展开：原点到每点累计和存表、递推式 3.31+四角查表式 3.32（pp.129–130，Figure 3.17 数值例 p.130） |
| tent function | 帐篷函数 | E023 | 分段线性三角帽=Bartlett 滤波（p.126） |
| bilinear kernel | 双线性核 | E023 | 两个线性 tent 的外积（p.126） |
| "Gaussian" kernel | 带引号高斯核 | E023 | tent⋆tent=cubic 近似样条，非精确采样高斯（p.126） |
| low-pass kernel | 低通核 | E023 | 放行低频衰减高频（p.126）；频响 §3.4 展开 |
| unsharp masking | 非锐化掩模 | E023 | 式 3.22 线性补差；暗房负片叠印=式 3.23（p.126） |
| transmittance | 透射率 | E023 | 式 3.23 负片叠印相乘（p.126） |
| negative ringing | 负值振铃 | E023 | 线性锐化产生负值（讲者术语，E023 实验 #8：3.22 min −0.60） |
| Sobel operator | 索贝尔算子 | E023 | 竖直 tent×水平中心差分，强调竖直边（Figure 3.14d 图在 p.125；构造句 pp.126–127） |
| central difference | 中心差分 | E023 | 导数正对像素而得名（p.126） |
| corner detector | 角点检测器 | E023 | 横竖二阶差分同现；对角边也响应（Figure 3.14e 图在 p.125；对角边句 p.127） |
| second derivative | 二阶差分 | E023 | 斜率的变化（p.127；「斜率的变化」为讲者白话） |
| distance transform | 距离变换 | E023 | Figure 3.11g（p.121）；§3.3 展开兑现于 E026：式 3.46–3.48 两度量与定义（p.139，释义句 i.e. 跨 p.140）、两趟光栅（Figure 3.23 p.140）、grassfire/骨架/signed（pp.140–141） |
| morphological filters | 形态学滤波 | E023 | Figure 3.11f（p.121）；§3.3 展开兑现于 E026：式 3.45 统一框架（p.138）与五操作（p.139；Figure 3.22 p.138） |
| shift invariance | 平移不变 | E023 | 式 3.17：输入平移则输出同样平移（p.122） |
| moving average | 滑动平均 | E023 | 方块核滤波的别名（p.125） |
| negative | 负片 | E023 | 暗房往事：模糊负片叠印锐化（p.126） |
| neighborhood filtering | 邻域滤波 | E023 | 线性滤波即最常用邻域算子（p.120） |
| separable filtering | 可分离滤波 | E023 | 预告首现；E024 展开：式 3.20 K=vhᵀ 外积、式 3.21 SVD 判据（p.124），两趟 2K vs K² 提速账（p.124）；Ex 3.10 习题本体 p.182 |
| band-pass filter | 带通滤波 | E023 | 预告首现；E024 展开：配方两步=先单位面积高斯平滑、再取一阶或二阶导数（式 3.24–3.26，p.127）；E021「band-pass」为频域视角先行词条 |
| steerable filter | 转向滤波 | E023 | 预告首现；E024 展开：方向导数=基函数线性组合（式 3.27–3.29，pp.127–128），一阶两基/二阶三基；特征描述子/边缘检测用途 p.129 |
| Laplacian pyramid | 拉普拉斯金字塔 | E023 | 提及；E030 展开（§3.5.3，p.155 起：Burt–Adelson 1983；带通=本层减上层插值回来 p.157；完全重构 p.157；塔里的「拉普拉斯」实为近似 DoG p.158） |
| border ring | 边缘环 | E023 | 讲者/实验术语：边界一圈像素的均值（本集实验 #4 口径） |
| outer product | 外积 | E024 | 两向量相乘得矩阵；可分离核=两个 1D 核的外积（式 3.20 K=vhᵀ，p.124） |
| rank-one | 秩一 | E024 | SVD 只剩第一个奇异值非零，整张核=一行乘一列（p.124；「一行乘一列」为讲者白话） |
| integral image | 积分图 | E024 | summed area table 的业界别名（本集 deck 主用叫法；原书用 summed area table，p.129） |
| Laplacian operator | 拉普拉斯算子 | E024 | 两个方向的二阶导相加（式 3.25，p.127） |
| Laplacian of Gaussian (LoG) | 高斯拉普拉斯 | E024 | 先平滑后拉普拉斯=等价直接卷「高斯的拉普拉斯」核（式 3.26，p.127）；小 σ 下与五点拉普拉斯相关 0.97 为讲者实测 |
| five-point Laplacian | 五点拉普拉斯 | E024 | 中心负、十字四正的紧凑 LoG 近似核（p.127） |
| scale space | 尺度空间 | E024 | σ 连续变大的一族平滑图，尺度即模糊程度（Witkin 1983，p.127；白话为讲者按） |
| oriented filter | 方向滤波器 | E024 | 与带通并列的一族（p.127「band-pass and oriented filters」）；steerable 属之 |
| feature descriptor | 特征描述子 | E024 | steerable 常用做特征描述子（§7.1.3 指向，p.129）；第 7 章展开 |
| recursive filtering | 递归滤波 | E024 | 输出依赖之前的输出（式 3.31，pp.130–131） |
| IIR (infinite impulse response) | 无限冲激响应 | E024 | 递归滤波的信号处理学名；脉冲响应永不归零（p.130） |
| FIR (finite impulse response) | 有限冲激响应 | E024 | 以前的核响应有限（p.130） |
| boxlets | 盒特征 | E024 | 人脸检测中正负相邻矩形拼的特征别名（Viola-Jones 2004，p.130） |
| directional derivative | 方向导数 | E024 | 沿单位向量 û 的导数（式 3.27，p.127） |
| median filter | 中值滤波 | E024 | 非线性滤波代表；取邻域中间值（p.131 首现预告；E025 展开：shot noise 滤除/不擅高斯噪声/排序取中位数，pp.132–133） |
| bilateral filter | 双边滤波 | E024 | 边缘保持平滑；对 shot noise 有失手处（p.131 首现预告，Figure 3.18 图注；E025 展开承诺兑现：式 3.34–3.37 双核乘积+失手程序对照，pp.133–137） |
| shot noise | 脉冲噪声 | E025 | 偶发极大值噪声；中值可滤、高斯摊成斑点（p.132；Figure 3.18e–h） |
| α-trimmed mean | α-截断均值 | E025 | 去掉最小/最大各 α 比例后取均值（pp.132–133；Figure 3.19b α-mean=4.6 复算） |
| weighted median | 加权中值 | E025 | 每像素按距离重复使用；式 3.33 p=1 最小化（p.133） |
| domain kernel | 域核 | E025 | 只看空间远近的核，式 3.35（p.134；Figure 3.19c 数值例 exp(−dist²/4) 复现） |
| range kernel | 值域核 | E025 | 只看灰度差的核，数据依赖，式 3.36（p.134；Figure 3.19d 为示意数值，e025_s2_numeric_notes.md 有读数结论） |
| anisotropic diffusion | 各向异性扩散 | E025 | 迭代自适应平滑=Perona–Malik 1990b 离散扩散方程（式 3.38–3.40，pp.135–136；无偏置收敛常数图警告 p.136） |
| guided image filtering | 引导滤波 | E025 | 权重由引导图决定；局部仿射模型+岭回归（式 3.41–3.43，pp.136–137；He, Sun, and Tang 2013） |
| bilateral grid | 双边网格 | E025 | 双边加速：高维 color/position 空间均匀网格降采样（p.134；Chen, Paris, and Durand 2007；E025 口播仅指向，细节原书为纲） |
| thresholding | 阈值化 | E026 | 二值化的来路：f≥t 记 1 否则 0，式 3.44（p.138；边界值 f=t 记 1，C1 断言） |
| structuring element | 结构元素 | E026 | 任意形状（3×3 方盒/圆盘/搜寻形状），式 3.45 的 S=其 1 像素数（pp.138–139）；「探针」为讲者比喻 |
| dilation | 膨胀 | E026 | 五操作之一：θ(c,1) |
| erosion | 腐蚀 | E026 | 五操作之一：θ(c,S) |
| majority | 多数票 | E026 | 五操作之一：θ(c,S/2) |
| opening | 开运算 | E026 | 先腐蚀再膨胀：去小物体留大区域（p.139；Figure 3.22 图注 p.138 原文 as it is not wide enough |
| closing | 闭运算 | E026 | 先膨胀再腐蚀：填小孔平滑边界（p.139；与 opening 极性对偶） |
| city-block distance | 城市街区距离 | E026 | d1=|k|+|l|，式 3.46 第一度量（p.139；别名 Manhattan distance；等值线呈菱形） |
| Euclidean distance | 欧氏距离 | E026 | d2=√(k²+l²)，式 3.47（p.139；d1(6,8)=14 vs d2=10，等值线呈圆） |
| vector-valued distance | 向量距离 | E026 | 精确欧氏距离变换存 (Δx,Δy) 按平方距离比较（p.140；原书未为向量版单独编式号；Danielsson 1980） |
| grassfire transform | 草火变换 | E026 | 距离变换别名：边界如火线烧进物体，距离=火烧到该点的时间（p.140，原书比喻） |
| chamfer | 倒角 | E026 | 距离变换别名，木工倒角；chamfer matching 倒角匹配=按距离图对齐（p.140） |
| skeleton | 骨架 | E026 | 距离变换的脊（p.140）；「烧到最后不灭的根」为讲者比喻（grassfire 语境） |
| medial axis transform | 中轴变换 | E026 | 骨架的学名 MAT（p.140；Tek and Kimia 2003） |
| signed distance transform | 带符号距离变换 | E026 | 原图与补图各算距离、一侧取负合并：正负号区分内外（pp.140–141；Lavallée and Szeliski 1995） |
| connected components | 连通域 | E026 | 同值相邻像素连成区域并打标签（pp.141–142；N4/N8 两种邻接；Haralick and Shapiro 1992 §2.3） |
| N4/N8 adjacency | N4/N8 邻接 | E026 | N4=水平垂直四邻、N8=加四对角（p.141 定义）；斜对角相接两块 N4 分裂/N8 合并（讲者实验演示） |
| second moments | 二阶矩 | E026 | 区域统计第 4 样：式 3.49 的 M 矩阵，特征值分析得长短轴朝向与长度（p.141） |
| Fourier transform | 傅里叶变换 | E027 | 逐频率登记增益与相位的账本；式 3.54 H(ω)=A·e^{jφ}、式 3.55 变换对记法（pp.143–144） |
| angular frequency | 角频率 | E027 | ω=2πf，式 3.50（p.142）；频响图横轴为归一化频率 f=ω/2π、满轴 0.5 |
| complex-valued sinusoid | 复正弦 | E027 | e^{jωx}=cos ωx+j·sin ωx（式 3.52，p.143）；过线性系统只乘一个复数（式 3.53） |
| phase shift | 相移 | E027 | Δφ=φo−φi=arg H(ω)，同频正弦探针测出（pp.142–143；Figure 3.24 语义） |
| frequency response | 频率响应 | E027 | H(ω)：每个频率各乘多少、挪多少相位；Table 3.1 六个小核的频响（p.145，程序重绘） |
| discrete Fourier transform (DFT) | 离散傅里叶变换 | E027 | 式 3.57 含 1/N 归一；k∈[−N/2,N/2] 有意义、更大 k 跟低频混叠（p.144） |
| fast Fourier transform (FFT) | 快速傅里叶变换 | E027 | O(N log₂N)：分 log₂N 级、每级 2×2 小变换加重排；分级图形像蝴蝶=butterfly 蝶形运算（p.144） |
| complex gain | 复数增益 | E027 | 讲者按：A·e^{jφ} 一个复数一并带走幅度与相位（式 3.53 语义；原书无此名） |
| two-dimensional Fourier transform | 二维傅里叶变换 | E028 | 一维公式与直觉直接搬进二维（p.146）；式 3.59 连续二重积分、式 3.60 离散 1/MN 归一（p.146）；实信号谱共轭对称=峰值成对出现 |
| oriented sinusoid | 有向正弦 | E028 | 式 3.58 s(x,y)=sin(ωx·x+ωy·y)（p.146）；不只横竖，还能斜着走的波（ωx, ωy 定方向） |
| power spectrum | 功率谱 | E028 | 式 3.61 ⟨[S(ωx,ωy)]²⟩=Ps(ωx,ωy)（p.146）：每个频率上的期望幅度平方；⟨·⟩=随机变量期望（E[·] 同义，脚注）；生成法=谱上填零均值高斯随机数再逆 FFT（p.146）；单张图看不出、平均才看得出（讲者实测 600 次蒙特卡洛） |
| Wiener filter | 维纳滤波 | E028 | 由图像与测量过程的统计模型推得的最优复原滤波（p.146–147）；二版删推导 |
| basis function | 基函数 | E028 | 展开信号的预制积木；DCT 用 ½ 像素偏移的余弦基（式 3.62，p.147；偏移使基系数对称，Wallace 1991）；与 E023 steerable「方向导数=基函数线性组合」同词异境 |
| Karhunen–Loève transform | K-L 变换 | E028 | 把信号去相关到最彻底→理论上引向最优压缩（p.148；「KL-」止于 p.147 行末跨页接 p.148「transform」）；需逐块现算 |
| principal component analysis (PCA) | 主成分分析 | E028 | K-L 分解可由图像 PCA 得到（p.147；Section 5.2.3 详述） |
| discrete cosine transform (DCT) | 离散余弦变换 | E028 | 傅里叶变换的变体、特别适合按块压缩（p.147）；式 3.62 一维点积、式 3.63 二维可分离实现 O(N log N)（pp.147–148）；K-L 的好近似=「近似最优」；广泛用于图像与视频压缩=JPEG 引擎（p.148） |
| peak signal-to-noise ratio (PSNR) | 峰值信噪比 | E028 | 式 2.120（定义处 p.101 补档；本集引用于 p.148）：峰值强度平方比均方误差取对数×10；八位图 Imax=255；适用前提=噪声人工加上、干净图已知（p.148）；讲者白话：差得越少分越高 |
| structural similarity (SSIM) | 结构相似度 | E028 | 感知相似度指标（Wang, Bovik et al. 2004，p.148）；量「人眼觉得像不像」非逐像素误差（讲者白话）；本集实测：全局与 PSNR 同向、纹理区单独算可反转（讲者示意） |
| total variation | 全变差 | E028 | 用 L1 范数的变分方法（p.148）；讲者白话：把去噪写成求最优的数学题，全变差加总的是相邻像素差的绝对值 |
| non-local means | 非局部均值 | E028 | §3.4.2 非线性滤波名单压轴（p.148；Buades, Coll, and Morel 2008）；本课程先不展开（讲者对课程安排的自述） |
| interpolation | 插值 | E029 | 放大=往样本之间填新样本（p.149–150）；式 3.64 把核里的 k、l 换成 r 乘 k、r 乘 l；插值核必须过原样本点；表 3.1 平滑核乘插值率 r 即成插值核（脚注 11 |
| upsampling | 上采样 | E029 | §3.5.1 标题的操作名（p.149 路线图）=放大分辨率；口播白话「往样本之间填点」 |
| downsampling | 降采样 | E029 | 缩小分辨率的统称（p.149 路线图）；脚注 13：decimation 在信号处理里泛指任何降采样或降率操作 |
| decimation | 抽稀 | E029 | §3.5.2 标题的操作名（p.153）；词源残酷=古罗马对怯懦部队十杀其一（脚注 13）；做法=先（概念上）低通滤波防混叠、再每隔 r 个样本取一个（式 3.66/3.67 同一核两用，讲者实测逐点相等最大误差为零） |
| polyphase | 多相 | E029 | 把核摆在输出格上、按相位分别取用的实现读法（p.150）；原书明言与直接叠加两种读法等价（讲者实测逐点一致，max err 0） |
| bicubic | 双三次 | E029 | 多数修图软件默认的插值核（p.151）；a 取负二分之一与负一两档（Figure 3.27）；a=−1 边缘更锐但过冲欠冲各约 0.09（讲者实测），a=−0.5 二次再现最大误差为零 |
| approximating kernel | 近似核 | E029 | 插值结果不穿过原样本点的核（p.150；立方 B 样条=tent 自卷积）；讲者白话：插值核像「必须踩过每个钉子」的线、近似核像「从钉子中间顺滑绕过去」的线 |
| ringing | 振铃 | E029 | windowed sinc 截断带来的波纹（pp.151–152）；2D 放大上半部可见周期性振铃波纹（讲者实测，目视）；有人反感振铃及其频率重复感 |
| spline | 样条 | E029 | 词源=制图车间画平滑曲线用的柔性木条或金属条（脚注 12，p.151）；计算机视觉用于弹性形变、散乱数据插值、运动估计与表面插值，多数图像处理都可在多分辨率框架用样条表示与操控（p.152）；称呼交叉申报：tent 自卷积所得三次样条按字面「½ 像素采样」复算对不上二项式核、对得上的是 tent⋆box 二次样条（讲者复算） |
| windowed sinc | 加窗的辛克函数 | E029 | 公认最高质量的插值器：保细节又避免混叠（p.152）；原书写明 raised-cosine（升余弦）窗（图 3.28 图注；Table 3.2 条目作 cosine-windowed）、未给图中连续曲线的窗宽参数；原书截断：图 3.27d 为 nine taps、Table 3.2 为 |n|≤4 即 9 抽头含两端零值（原文未说明是否同一窗口）；讲者口径 cos²(πx/5)、|x|≤2.5 在 r=2 下 11 抽头（申报差异） |
| binomial kernel | 二项式核 | E029 | Burt–Adelson 1983a，[1,4,6,4,1]/16（Table 3.2，p.154）；对高低频分离 does a decent job 但仍留不少高频细节→缩小后可能混叠；对图像融合这类应用这点混叠无关紧要（p.154）；本集实测：|H(0.45π)| 0.334 七者最小（cuts off a lot of frequencies）、混叠能量七者最小、PSNR 17.92 七者最低但与最优差 0.29 dB |
| QMF-9 | Q-M-F-九 | E029 | Simoncelli–Adelson 1990b（Table 3.2，p.154）；用于小波去噪、aliases a fair amount（混叠不少）；原系数按 √2 增益归一、可自逆（self-inverting）；讲者实测与自逆条件一致：H(π/2)·√2=1.0001 且全系数 2Σc²=0.9994（点检验+均值检验） |
| JPEG 2000 9/7 | J-P-E-G-2000-九七 | E029 | Taubman–Marcellin 2002 的 9/7 分析滤波器（Table 3.2，p.154 列出）；原书对 r=2 低倍率要求 more careful filter design（p.154）；本集实测 PSNR 18.21 榜首但与 linear 只差 0.05 dB |
| Hann | 汉恩 | E084 | 汉恩窗（Hann window）：相位相关前加升余弦窗抑制边界十字亮线；本集口播「Hann 窗」读汉恩窗，加窗效果为 lab 实测（峰值干净、位置稳） |
| raised-cosine window | 升余弦窗 | E029 | 图 3.28 图注的窗函数名（p.152）；形状即 cos²（讲者口径 cos²(πx/5) 对应） |
| image pyramid | 图像金字塔 | E030 | E028 移交首提；多分辨率表示总纲：底层原图、往上一层比一层小（p.155）；用途=由粗到细搜索、多分辨率融合、图形软硬件分数级缩小（MIP-map）（pp.154–155） |
| Gaussian pyramid | 高斯金字塔 | E030 | 搭法=先低通模糊再隔二取样存上一层（p.155）；相邻层采样率差一倍=倍频程金字塔；核取五抽头 a=3/8（式 3.68）即二项式核 [1,4,6,4,1]/16（式 3.69，与 E029 Table 3.2 全等）；名字由来=反复卷积收敛到高斯（脚注 14：任何平滑核都如此，收敛率未给 |
| octave | 倍频程 | E030 | 相邻层采样率差一倍为一级（p.155 octave pyramid）；64 倍尺度范围整倍频程 6 级、半倍频程 12 级（讲者实测，阶梯步数口径） |
| half-octave pyramid | 半倍频程金字塔 | E030 | Crowley–Stern 1984 引入视觉界、名 DOLP（p.159）；相邻层尺度变化小、作者声称由粗到细算法表现更好；特征检测常用半倍频程甚至四分之一倍频程，但抽稀只在每个整倍频程处做（p.159） |
| quincunx sampling | 梅花采样 | E030 | 半倍频程加梅花五点棋盘采样格的专名（p.159；引 Feilner, Van De Ville, and Unser 2005）；原书未给滤波器组系数 |
| DOLP | 低通之差变换 | E030 | difference of low-pass，半倍频程金字塔的原名（p.159，Crowley and Stern 1984） |
| MIP-map | MIP 贴图 | E030 | 图形软硬件用金字塔做分数级缩小的那套东西（p.155 开篇提及）；E032 展开：Williams 1983 纹理映射预滤波（p.172）、脚注 16「multi in parvo=很多放进一个里」、r 估计与 trilinear（p.173，词条另列） |
| difference of Gaussian (DoG) | 高斯之差 | E030 | 塔里的「拉普拉斯」实为近似 DoG：两个不同宽度的高斯各自平滑图像后相减（式 3.70，p.158）；空域中心正两侧负、频域直流处归零（讲者实测重绘）；与同尺度数值 LoG 剖面 |cc|=0.9184 反相（原书未给符号约定，只论形状同族不作等式断言 |
| perfect reconstruction | 完全重构 | E030 | 拉普拉斯金字塔性质：全部带通图+顶层低通图一丝不差还原原图（p.157「exactly reconstruct」）；讲者实测最大误差 1.11e−16（机器精度）；前提=重构端系数为分析系数两倍（p.156 Figure 3.32 图注直接给出未推导；不乘二误差 0.793）；变体（低通取自原图模糊版）混叠更少但不再自逆（err=0.353） |
| analysis / synthesis | 分析 / 合成 | E030 | Figure 3.32 信号处理框图两路的命名（p.156）：分析=模糊+降采样下行；合成=插零上采样+滤波+相加重构；两路运算几乎一样，合成端系数×2 |
| coarse-to-fine | 由粗到细 | E030 | 金字塔首要用途（p.154–155）：在不同尺度上找目标或图案；半倍频程作者声称由粗到细算法表现更好（p.159） |
| shifts and adds | 移位和加 | E030 | 二项式核的实现红利（p.155）：五系数皆十六分之几、除以十六只是移位 |
| local Laplacian filtering | 局部拉普拉斯滤波 | E030 | Paris, Hasinoff, and Kautz 2011：在塔构造里做局部范围裁剪，实现保边缘滤波与色调映射（p.158–159）；Aubry 等人 2014 加速并做出风格迁移 |
| tone mapping | 色调映射 | E030 | 高动态范围图像压到显示域的操作（p.158–159 与局部拉普拉斯滤波同段提及；详见原书 Section 10.2.1）；E011 历史课曾一笔带过 |
| DC (direct current) | 直流分量 | E028 | E028 白话定义：块内平均值/零频（JPEG 每块 DC 系数）；E030 白话「整幅图的平均亮度」，DoG 频域在直流处恰好归零（式 3.70 性质） |
| scale-space | 尺度空间 | E024 | E024 ：尺度就是模糊程度（LoG 有很好的尺度空间性质）；E030 「尺度空间社区」研究高斯-拉普拉斯的数学性质（p.158） |
| wavelet | 小波 | E031 | §3.5.4 正课（p.159 下起）：账本视角=分解后系数不多不少（紧框架）；构造=拆奇偶+可逆滤波逐步抬性能（p.160–163） |
| tight frame | 紧框架 | E031 | 分解后系数总数与像素一样多、账是平的（p.160）；E030 预埋「多出来的账」在本集得名；对比拉普拉斯金字塔多出 1/3（讲者实测 1.3320） |
| overcomplete | 超完备 | E031 | 分解后系数比像素多（p.160）；与紧框架一笔账分清：紧框架账平、超完备多账，两者都不是丢信息 |
| HH | 高高带 | E031 | high–high，高通加高通的方向细节带（p.161）；只有水平+垂直混合的能量才进 HH（讲者实测横条纹 HH=0） |
| HL | 高低带 | E031 | high–low，先行高通、再列低通（按 Figure 3.37b 次序，p.161）；竖条纹能量集中于此（实测 43.9 vs 0 对调） |
| LH | 低高带 | E031 | low–high，先行低通、再列高通（p.161）；横条纹能量集中于此（实测 43.9） |
| lifting | 提升 | E031 | 提升小波构造（p.162–163）：拆奇偶→减均值预测→加回修正，每步平凡可逆（讲者实测拆开再装回误差 1e−16 量级）；有效低通核系数 [−1/8, 1/4, 3/4, 1/4, −1/8] 时 H(π)=0 |
| quadrature mirror filter (QMF) | 正交镜像滤波器 | E031 | 两只滤波器频响以四分之一采样率为轴互为镜像（H₁(ω)=H₀(ω+π)，p.162）；「正交」译自 quadrature，指这个四分之一处；E029 Table 3.2 的 QMF-9 为其具体系数（词条另列） |
| aliasing | 混叠 | E020（使用）/E031（白话定义） | 高频冒充低频（E031 白话：抽稀没滤波时发生）；原书脚注 15：信号慢慢平移时按帧率采样就能看到（p.164）；E029 抽稀词条的「防混叠」即此 |
| Nyquist frequency | 奈奎斯特频率 | E031 | 采样率定了，能装下的最高频率=采样率的一半（讲者白话，p.164）；原书「低于奈奎斯特频率采样」=采样太稀、跟不上最高频成分的两倍→混叠 |
| second-generation wavelets | 第二代小波 | E031 | 提升小波的通称（p.164）：得名于能轻松适应不规则采样拓扑（如图形学多分辨率曲面编辑）；加权提升小波还可做低层图像操作与第四章优化类算法的稀疏线性方程组预处理（p.164） |
| steerable pyramid | 可导向金字塔 | E031 | 方向可控的过完备变换，提出者给它起了个长名字（p.164–165）；过完备表示消除了混叠问题、分析基与合成基相同=自逆（p.165）；本集只示意频带划分方式、不实现完整滤波器组（口径申报在案） |
| splining | 样条融合 | E031 | Burt 和 Adelson (1983b) 对金字塔融合工序的命名（p.165）；样条=绘图桌上弯出光滑曲线的弹性细条（E029 词源条目），此处即平滑过渡函数 |
| ghosting | 重影 | E031 | 两图纹理叠在一起时直接交叉渐变产生的融合失败痕迹（p.165「"ghosting" effects when two textures are overlaid」）；金字塔融合的动机即避开它 |
| geometric transformation (image warping) | 几何变换（图像扭曲） | E031（移交首提）/E032（正课） | 图像扭曲改的是图像函数的定义域而非值域（Figure 3.43 图注，p.168）；点算子/滤波改「每个位置上的值」，几何变换改「值从哪里取」 |
| forward warping | 前向扭曲 | E032 | 源像素贴到目标位置（Figure 3.45，p.170）；放大 1.5× 时最近邻洞 1185/2209=53.6%（讲者实测）；贴过去有洞有重叠 |
| inverse warping | 逆向扭曲 | E032 | 对目标图每像素算出源图对应位置、从源图采样（Figure 3.46，p.170–171）；无洞、非整数位置重采样可用高质量滤波器；抽回来无缝 |
| splatting | 溅泼法 | E032 | 前向扭曲的软化版：值按双线性权重摊给四个邻居、记好权重最后归一（p.170）；原书自评仍有中等走样和相当多的模糊（讲者实测共同有效区 RMSE 0.0381 vs 逆向双线性 0.0064）；本集可见层用英文原词 |
| resampling | 重采样 | E032 | 非整数位置的值用 FIR 重采样滤波核加权求和（式 3.75，p.172）；E029 插值即其在整数栅格的特例；抽稀时核按抽稀率拉伸，一般变换下采样率没法定义（横向拉伸纵向压扁） |
| trilinear MIP-mapping | 三线性 MIP-map | E032 | 先按层数小数部分线性混合上下两层、层内再双线性（p.173）；层号 l=log2 r，r 取矩阵 A 元素绝对值的最大值（抑走样）或最小值（减模糊）；MIP-map 词条（E030 预告「后面再讲」）在本集展开 |
| anisotropic texture filtering | 各向异性纹理过滤 | E032 | 缩小时采样区域随图像局部走向拉长、核跟着对齐（p.174）；与 E025 各向异性扩散（Perona–Malik 平滑）同名不同物 |
| EWA (elliptical weighted average) | 椭圆加权平均 | E032 | 各向异性过滤的最优做法（p.174–175）：逆协方差定椭圆形状、按拟合法加权平均；作者第一人称自评「尽管有高质量的名声……他们在自己的工作里发现质量常常更差而且还慢」（L317–322） |
| mesh-based warping | 网格变形 | E032 | 局部变形路线总称（p.175–177）：稀疏对应点加变形网格/更密对应点/有向线段对应/均匀四边形网格四法（Figure 3.49 图注）；参数化变换参数少、全局一致，局部变形要多得多的自由度 |
| Beier–Neely | Beier–Neely（人名） | E032 | 有向线段对应的提出者（1992，p.176–177）：线段上像素按指定精确迁移、其余平滑插值；每对线段为附近像素指定平移加旋转加缩放（相似变换）；权重段内取垂距、段外取到端点的较短距离（原书未给公式常数 |
| morphing | 图像变形 | E032 | 扭曲加混合两幅图（p.177–178）：先把每幅图向对方扭曲、再按 1−t 与 t 混合，位移线性插值；对应关系设得好特征对齐重影就不会出现（讲者实测重影带 1490→488，缩到三分之一）；运动离线性很远时「部分扭向目标」要小心定义 |
| cross-dissolve | 交叉淡化 | E032 | 两图直接按权重混合（p.178）；不先扭曲对齐就出重影（Figure 3.51 顶行），morphing=先对齐再淡化的对照 |
| texture mapping | 纹理映射 | E033（原书行文在 E032 范围 p.168） | 图形学对图像变形的叫法：把图像贴到曲面（p.168，E032 讲几何变换时在该页起）；E033 白话回指并顺势接 MIP-map |
| coring | coring（不译） | E034 | 原书 Ex 3.21「wavelet-based coring and denoising」（p.186）：小波细节带小系数用软阈值压掉、低通带原样保留；口播直呼英文并当场解释（「把小系数用分段线性的软阈值压掉」），中文不译为全课程口径 |
| soft threshold | 软阈值 | E034 | 分段线性压噪函数：系数绝对值低于阈值压到零、其余按线性缩小（p.186 Ex 3.21 本集实现口径）；区别于硬阈值（直接置零） |
| total least squares | 全最小二乘 | E036 | 原书 A.2.1（p.929–930）：测量误差四面八方都有（变量含误差模型）时最小化齐次平方误差 Σ(ax+by+c)²，解=AᵀA 最小特征值对应特征向量=最后右奇异向量；与竖直量距 VLS（式 A.25）对照 |
| design matrix | 设计矩阵 | E036 | 原书 p.927 脚注 5 语境：一般线性最小二乘 ‖Ax−b‖² 里 A 每行装一个数据点（口播白话「一行装一个数据点」）；注意与直线拟合的横轴 x 区分（原书脚注 5 专门提醒） |
| errors-in-variables | 变量含误差模型 | E036 | 原书 p.929：测量点各方向都有不确定度（uncertainty in all directions）的模型；译名按 D28 定「变量含误差模型」，不译「全变量误差模型」防与「全最小二乘」混淆 |
| robust statistics | 鲁棒统计 | E037 | 附录 B.3 主题（p.945 起）：数据不服从理想假设（含外点）时仍能可靠估计的方法族；E037 双锚之一（另一处=第 4 章 §4.1.3） |
| outlier | 外点 | E037 | 原书 p.946：测量过程的大故障（gross failures） |
| M-estimation | M-估计 | E037 | 原书 p.946：给每条残差套罚函数 ρ 再求和的目标（式 B.21），统计文献通称（Huber 1981 等）；与 E003 残差词条衔接 |
| robust loss function | 鲁棒损失函数 | E037 | 原书 p.202（式 4.15）与 p.203（Barron 2019 一族）：对大误差宽容的罚函数尺子；α=2 退化为平方 |
| shape parameter | 形状参数 | E037 | 式 B.21 的 α：管损失有多鲁棒（p.946）/外点残差打几折（p.203 Barron 表述）；E037 口播白话「对外点多狠」 |
| scale parameter | 尺度参数 | E037 | 式 B.21 的 c：管碗口多宽、多大以内的残差算自己人（p.946/p.203）；E037 实验交给 MAD×1.4 每轮自调 |
| Charbonnier loss | 沙博尼耶损失 | E037 | α=1 退化（图 4.7 题注）：小残差按平方、大残差按绝对值记账（p.202）；中文音译「沙博尼耶」为全课程口径（E037 起） |
| Cauchy loss | 柯西损失 | E037 | α=0 退化（题注）：影响函数有界、大残差权重趋零；E037 主实验尺子（17 轮收敛 0.4987） |
| Geman-McClure loss | 杰曼-麦克卢尔损失 | E037 | α=−2 退化（题注）：E037 读数里最狠的一把（r=5 时代价 1.72）；人名音译照口播 |
| Welsch loss | 韦尔施损失 | E037 | α=−1 退化（题注口径，D30：与同图图例 α=−∞ 不一致按题注）；E037 E5 网格实验尺子（更狠的一把） |
| influence function | 影响函数 | E037 | ψ(r)=ρ′(r)（式 B.23，p.946）：一个点拽答案的手劲；平方损失手劲线性加码、柯西型有界；除以残差得权重函数 |
| weight function | 权重函数 | E037 | w(r)=ψ(r)/r（p.946）：手劲折算成每单位误差的权重；IRLS 每轮重算的就是它 |
| iteratively reweighted least squares (IRLS) | 迭代重加权最小二乘 | E037 | 式 B.24+两步轮换（p.947）：固定权重解加权最小二乘→拿新残差重算权重；把非二次损失拆成一串能直接套最小二乘的小问题；E036 加权最小二乘（B.16–B.20）的鲁棒版 |
| median absolute deviation (MAD) | 中位数绝对偏差 | E037 | 式 B.25（p.947）：残差绝对值取正中间那个、乘 1.4=内点噪声标准差的鲁棒估计（×1.4 照原书原文，非通用 1.4826）；中位数一两个大数拽不动 |
| least median of squares (LMS) | 最小中位数法 | E038 | 原书 p.511：并列那招 |
| inlier | 内点 | E036 | 原书 p.511：残差落在门槛 ϵ 内的对应点；RANSAC 验线数的就是它，票数=内点计数 |
| Preemptive RANSAC | 抢占式 RANSAC | E038 | 原书 p.512 变体族：提前逐出表现差的候选，加速用 |
| PROSAC | 渐进抽样一致 | E038 | 原书 p.512 变体族：按匹配质量排序渐进抽样，加速用 |
| MLESAC | 极大似然抽样一致 | E039 | 原书 p.511 起变体族列举（）：MLESAC、可微的 DSAC、Graph-Cut RANSAC、MAGSAC、ESAC |
| DSAC | 可微抽样一致 | E039 | 原书 p.511 起变体族：可微的 DSAC |
| Graph-Cut RANSAC | 图割 RANSAC | E039 | 原书 p.511 起变体族列举（）；机制不展开 |
| MAGSAC | MAGSAC（专名） | E039 | 原书 p.511 起变体族列举（）；机制不展开 |
| ESAC | ESAC（专名） | E039 | 原书 p.511 起变体族列举（）；机制不展开 |
| GNC | 渐进非凸 | E039 | 原书 p.512：渐进非凸的 GNC 在不少几何任务上能胜过 RANSAC |
| information matrix | 信息矩阵 | E040 | 原书 p.952：逆协方差矩阵的别名，这种简单情形下等于 Hessian；E040 lab 按 ½ 负对数似然约定直验（E1a） |
| Cramér–Rao lower bound | 克拉默-拉奥下界 | E040 | 原书 p.952：后验能量非二次型时，最优解附近估 Hessian 取逆=不确定度下界；协方差不会比这个局部二次近似算出的更小（实际≥下界，线性高斯等号成立） |
| posterior covariance | 后验协方差 | E040 | 式 B.19（p.945）：Σ=C⁻¹=(Σᵢ HᵢᵀCᵢHᵢ)⁻¹ |
| dominant modes of uncertainty | 主导不确定方向 | E040 | 原书 p.952：协方差太大存不下时，特征值分析找主导晃动方向（E040 实测 cos=0.9930 绕质心转线） |
| damped least squares | 阻尼最小二乘 | E041 | 原书 p.931：加对角阻尼项的高斯-牛顿（式 A.48–A.50） |
| Levenberg–Marquardt | 列文伯格-马夸特（LM） | E041 | 原书 p.931：牛顿一阶近似+自适应阻尼 λ 的合称（括注 Levenberg 1944/Marquardt 1963） |
| trust region | 信任域 | E041 | 原书 p.931：LM 是更一般的信任域方法的一个例子 |
| neuron (unit) | 神经元 | E042 | 原书 p.270：加权和+非线性激活的最小计算单元（式 5.45–5.46） |
| weight | 权重 | E042 | 原书 p.270：可学习的乘法系数（式 5.45 的 wi） |
| bias | 偏置 | E042 | 原书 p.270：式 5.45 的 bi |
| activation function | 激活函数 | E042 | 原书 p.270：加权和之后的非线性重映射 h（式 5.46） |
| ReLU (Rectified Linear Unit) | 整流线性单元 | E042 | 原书 p.272：h(y)=max(0,y)（式 5.49，Nair-Hinton 2010/Krizhevsky 2012 起） |
| perceptron | 感知器 | E042 | 原书 p.271：最早的神经元单元（Rosenblatt 1958，Figure 5.23a 权重显式画法） |
| fully connected (FC) layer | 全连接层 | E042 | 原书 p.271：满权重矩阵的层（式 5.47 sl=Wlxl） |
| multi-layer perceptron (MLP) | 多层感知器 | E042 | 原书 p.272：只由全连接层（无卷积）组成的网络 |
| Grad-CAM (gradient-weighted class activation mapping) | 梯度加权类激活映射 | E049 | 原书 p.308：Selvaraju et al. 2017 命名（与 saliency maps 2013/activation maximization 2016 同族） |
| self-supervised learning | 自监督学习 | E049 | 原书 p.313：pretext task 标签自动派生（对比学习谱系 memory bank→MoCo→SimCLR→BYOL/SimSiam；CLIP 400M pairs 收束） |
| recurrent neural network (RNN) | 循环神经网络 | E050 | 原书 p.321：把上一时刻提取的特征喂给下一时刻的处理（Such architectures are called Recurrent Neural Networks） |
| LSTM (long short-term memory) | 长短期记忆网络 | E050 | 原书 p.321–322：Hochreiter and Schmidhuber 1997 |
| GRU (gated recurrent unit) | 门控循环单元 | E050 | 原书 p.321–322：与 LSTM 并列的门控单元 |
| 3D convolution | 三维卷积 | E050 | 原书 p.317–318：感受野盒子在 (x, y, c) 之外沿时间再开一维 |
| C3D | 三维卷积网络 C3D | E050 | 原书 p.318：Tran, Bourdev et al. 2015 |
| attention (mechanism) | 注意力 | E050 | 原书 p.322 末预告（5.5.3 主讲述=E051）：self-attention/visual attention/一步并行整序列 |
| mesh (geometry) | 三角网格 | E050 | 原书 p.320：3D point clouds and meshes 小节 |
| ViT | 视觉 Transformer | E051 | 原书 p.326–327：Dosovitskiy, Beyer et al. 2021 命名 Vision Transformer 简称 ViT |
| DETR | 检测 Transformer | E051 | 原书 p.327：Carion, Massa et al. 2020 |
| VQ-VAE | 向量量化变分自编码器 | E051 | 原书 p.328–336（5.5.4=E052 主讲述）：van den Oord et al. 2017 |
| GAN | 生成对抗网络 | E052 | 原书 p.331：Goodfellow et al. 2014 极小极大博弈（式 5.79） |
| VAE | 变分自编码器 | E052 | 原书 p.329：Kingma and Welling |
| DCGAN | 深度卷积生成对抗网络 | E052 | 原书 p.332–333：Radford et al. |
| LAPGAN | 拉普拉斯金字塔生成对抗网络 | E052 | 原书 p.332：Denton et al. |
| cGAN | 条件生成对抗网络 | E052 | 原书 p.333：Mirza and Osindero |
| InfoGAN | 信息最大化生成对抗网络 | E052 | 原书 p.333：Chen et al. 加互信息项 |
| CycleGAN | 循环一致生成对抗网络 | E052 | 原书 p.334：Zhu et al. |
| pix2pix | 图到图翻译网络 | E052 | 原书 p.333：Isola et al. |
| LeCun | 勒昆 | E053 | 原书 p.337：LeCun, Bottou et al. 1998（Ex 5.2「LeNet」架构出处） |
| histogram of oriented gradients (HOG) | 方向梯度直方图 | E060 | 原书 p.376–377：Dalal–Triggs 2005 行人检测主特征 |
| support vector machine (SVM) | 支持向量机 | E048（首次使用）/E060（展开） | 原书 p.374：Osuna et al. 1997 用于人脸检测 |
| Boosting | 提升方法 | E060 | 原书 p.374：把许多弱学习器加权组合成强分类器 |
| AdaBoost | 自适应提升 | E060 | 原书 p.375：Freund–Schapire 的 Boosting 具体算法（弱学习器加权投票、符号定类别） |
| deformable part model (DPM) | 可变形部件模型 | E060 | 原书 p.377：Felzenszwalb et al. 在 HOG 行人检测器上加可变形部件 |
| DIFS (distance in feature space) | 特征空间内距离 | E060 | 原书 p.373：Sung–Poggio 每簇两条马氏距离之一 |
| DFFS (distance from feature space) | 到特征空间的距离 | E060 | 原书 p.373：另一条马氏距离（投影残差） |
| active appearance model (AAM) | 主动外观模型 | E059（展开）/E060（回指） | 原书 p.365–366 展开（E059）、p.372 作模板法例（E060） |
| HOG | 方向梯度直方图 | E060 | 裸键行（全短语键「histogram of oriented gradients (HOG)」只匹配全称；本集口播 8 处裸 HOG 需读英文音 hog，与 SIFT 读法同例） |
| SVM | 支持向量机 | E048（首次使用）/E060（展开） | 裸键行（全短语键只匹配全称；本集口播 6 处裸 SVM 需字母读法） |
| AAM | 主动外观模型 | E059（展开）/E060（回指） | 裸键行（本集口播 1 处裸 AAM 需字母读法；全短语键行见上） |
| DPM | 可变形部件模型 | E060 | 裸键行（本集口播 1 处裸 DPM 需字母读法；全短语键行见上） |
| DIFS | 特征空间内距离 | E060 | 裸键行（本集口播 2 处裸 DIFS 需字母读法；全短语键行见上） |
| DFFS | 到特征空间的距离 | E060 | 裸键行（本集口播 2 处裸 DFFS 需字母读法；全短语键行见上） |
| intersection over union (IoU) | 交并比 | E061 | 原书 p.380：预测框与真值框交集÷并集（式 6.6） |
| IoU | 交并比 | E061 | 裸键行（口播裸 IoU 三处；全短语键行见上） |
| average precision (AP) | 平均精度 | E061 | 原书 p.381：P-R 曲线下面积（Figure 6.27a） |
| AP | 平均精度 | E061 | 裸键行（口播裸 AP 多处；全短语键行见上） |
| mean average precision (mAP) | 平均精度均值 | E061 | 原书 p.381：各类 AP 再平均；VOC 单阈值 0.5、COCO 0.50–0.95 |
| mAP | 平均精度均值 | E061 | 裸键行（口播裸 mAP 两处；全短语键行见上） |
| non-maximal suppression (NMS) | 非极大值抑制 | E061 | 原书 p.380：置信度贪心去重叠（口播 3 处，读字母 N-M-S） |
| NMS | 非极大值抑制 | E061 | 裸键行（口播裸 NMS 三处；全短语键行见上） |
| region proposal network (RPN) | 区域提议网络 | E061 | 原书 p.382：替代 selective search 的卷积提议网络（口播 3 处，读字母 R-P-N） |
| RPN | 区域提议网络 | E061 | 裸键行（口播裸 RPN 三处；全短语键行见上） |
| feature pyramid network (FPN) | 特征金字塔网络 | E061 | 原书 p.383：自顶向下把高层语义灌回高分辨率层（口播 4 处，读字母 F-P-N） |
| FPN | 特征金字塔网络 | E061 | 裸键行（口播裸 FPN 四处；全短语键行见上） |
| single shot multibox detector (SSD) | 单发多框检测器 | E061 | 原书 p.384：单阶段检测器代表（口播 1 处，读字母 S-S-D） |
| SSD | 单发多框检测器 | E061 | 裸键行（口播裸 SSD 一处；全短语键行见上）；**E083 起运动估计的 SSD 指平方差之和，见下条** |
| R-CNN | 区域卷积网络 | E061 | 原书 p.382：Girshick 等 2014 区域提议+CNN+SVM（口播 4 处；裸键行，读字母 R-C-N-N |
| Fast R-CNN | 快速区域卷积网络 | E061 | 原书 p.382：换序+FC 双头、复用卷积（口播 2 处；读 Fast R-C-N-N） |
| Faster R-CNN | 更快区域卷积网络 | E061 | 原书 p.382：RPN 替代 selective search（口播 1 处；读 Faster R-C-N-N） |
| YOLO | 只看一次检测器 | E061 | 原书 p.384：You Only Look Once 单阶段家族（口播 4 处，读英文音 YOLO） |
| YOLOv4 | 第四版 YOLO | E061 | 原书 p.385：Bochkovskiy 等 2020，neck+两类技巧（口播 1 处，读 YOLO-v-four） |
| RetinaNet | 视网膜网络 | E061 | 原书 p.384：FPN 上单阶段+焦点损失（口播 1 处，读 Retina-Net） |
| VOC | 视觉对象类挑战赛 | E061 | 原书 p.379：PASCAL VOC 二十类、分类+检测双赛道（口播 4 处，读字母 V-O-C） |
| COCO | 上下文常见物体数据集 | E061 | 原书 p.379：每图多物体+像素级分割（口播 4 处，读英文音 COCO） |
| ILSVRC | 大规模视觉识别挑战赛 | E061 | 原书 p.379：2010 发布，2 万→140 万图、20→1000 类（口播 1 处，读字母） |
| PDQ | 基于概率的检测质量 | E061 | 原书 p.381：probability-based detection quality（口播 1 处，读字母） |
| LVIS | 大词表实例分割数据集 | E061 | 原书 p.385：Large Vocabulary Instance Segmentation（口播 1 处，读字母） |
| selective search | 选择性搜索 | E061 | 原书 p.382：R-CNN 抽约 2000 区域提议的算法（口播 2 处，读英文音） |
| Jaccard | 雅卡尔 | E061 | 原书 p.380：IoU 别名 Jaccard 指数/相似系数（口播 2 处，读英文音） |
| AlexNet | 亚历克斯网络 | E061 | 原书 p.382：R-CNN 可选骨干之一（口播 1 处，读 Alex-Net） |
| VGG | V-G-G 网络 | E061 | 原书 p.382：R-CNN 可选骨干之一（口播 1 处，读字母） |
| ResNet | 残差网络 | E061 | 原书 p.383：DETR 骨干（口播 1 处，读 Res-Net） |
| EfficientDet | 高效检测器 | E061 | 原书 p.385：YOLOv4 对照快检测器（口播 1 处，读 Efficient-Det） |
| TensorFlow | 张量流 | E061 | 原书 p.385：TensorFlow Object Detection API（口播 1 处，读 Tensor-Flow） |
| PyTorch | Py-Torch | E061 | 原书 p.385：Detectron2 所属框架（口播 1 处，读 Py-Torch） |
| FCN (fully convolutional network) | 全卷积网络 | E062 | 原书 p.388：单网络逐像素标注（口播 2 处，读字母 F-C-N） |
| U-Net | U 形网络 | E062 | 原书 p.388：细层跳连提升分辨率（口播 3 处，读英文音 U-Net） |
| CRF (conditional random field) | 条件随机场 | E062 | 原书 p.387：能量最小化/贝叶斯推断的经典框架（口播 4 处，读字母 C-R-F） |
| TextonBoost | 纹理基元提升 | E062 | 原书 p.387：Shotton 等 2009 一元/成对势分割系统（口播 2 处，读英文音） |
| LayoutCRF | 布局条件随机场 | E062 | 原书 p.388：TextonBoost 的 CRF 扩展（多实例/遮挡/三维模型；口播 1 处） |
| PSPNet | 金字塔场景解析网络 | E062 | 原书 p.388：空间金字塔池化聚合多尺度特征（口播 1 处） |
| UPerNet | 统一感知解析网络 | E062 | 原书 p.388–389：FPN+金字塔池化多头（场景/物体/部件/材质/纹理；口播 2 处） |
| HRNet | 高分辨率网络 | E062 | 原书 p.389：全程保留高分辨率、跨分辨率交换信息（口播 1 处） |
| Mask R-CNN | 掩码区域卷积网络 | E062 | 原书 p.391–392：Faster R-CNN+RPN+掩码分支（口播 4 处；读 Mask R-C-N-N） |
| PQ (panoptic quality) | 全景质量 | E062 | 原书 p.393：同时平衡实例与语义两任务的指标（口播 1 处，读字母 P-Q） |
| OpenPose | 开放姿态 | E062 | 原书 p.396：实时多人二维姿态系统（口播 1 处，读 Open-Pose） |
| DensePose | 密集姿态 | E062 | 原书 p.396：人体每像素对应三维表面点（口播 2 处，读 Dense-Pose） |
| Boundary IoU | 边界交并比 | E062 | 原书 p.392：替代 Mask IoU 的边界指标（口播 2 处，读 Boundary I-O-U） |
| snakes | 活动轮廓 | E062 | 原书 p.392：随时间演化的闭合轮廓（口播 1 处，读英文音） |
| gist | 场景概要 | E062 | 原书 p.394：颜色与构图的全局描述（口播 1 处，读英文音） |
| pop-up | 立体弹出 | E062 | 原书 p.395：照片沿边界折成平面模型（口播 1 处，读英文音） |
| texton | 纹理基元 | E062 | 原书 p.387：滤波器响应聚类得到的纹理类（口播 1 处，读英文音） |
| CLIP | 对比语言-图像预训练 | E063 | 原书 p.403：用图文对比学习做零样本分类（口播 9 处，读 C-L-I-P） |
| VQA | 视觉问答 | E063 | 原书 p.407–408：给图与问题生成答案（口播 6 处，读 V-Q-A） |
| BLEU | 双语评估替补 | E063 | 原书 p.405：描述/翻译指标（口播 2 处，读 B-L-E-U） |
| ROUGE | 面向召回的要点评估 | E063 | 原书 p.405：描述/摘要指标（口播 2 处，读 R-O-U-G-E） |
| METEOR | 带显式词序的翻译评估 | E063 | 原书 p.405：描述/翻译指标（口播 2 处，读 METEOR） |
| CIDEr | 基于共识的图像描述评估 | E063 | 原书 p.405–406：多参考共识指标（口播 2 处，读 C-I-D-E-R） |
| SPICE | 语义命题图像描述评估 | E063 | 原书 p.406：语义命题指标（口播 3 处，读 S-P-I-C-E） |
| SlowFast | 慢快双路网络 | E063 | 原书 p.399：慢路低频+快路高频（口播 3 处，读 Slow-Fast） |
| token | 词元 | E063 | 原书 p.403/407：文本与图像的离散单元（口播 2 处，读英文音） |
| BabyTalk | 婴儿说话描述系统 | E063 | 原书 p.400：检测物体属性关系再生成描述（口播 1 处，读 Baby-Talk） |
| Decoder | 解码器 | E063 | 原书 p.407：DALL·E 的解码 Transformer（口播 3 处，读 Decoder） |
| VQ-VAE-2 | 向量量化变分自编码器第二代 | E063 | 原书 p.407：DALL·E 的图像 token 编码器（口播 1 处，读 V-Q-VAE-2） |
| ViLBERT | 视觉-语言 BERT | E063 | 原书 p.409：视觉-语言预训练代表（口播 1 处，读 ViLBERT） |
| Oscar | 视觉-语言预训练模型 | E063 | 原书 p.409：视觉-语言预训练代表（口播 1 处，读 Oscar） |
| ResNet-50 | 残差网络 50 层 | E063 | 原书 p.403：CLIP 的对比基准（口播 1 处，读 Res-Net-50） |
| VisDial | 视觉对话数据集 | E063 | 原书 p.409：多轮视觉对话数据（口播 1 处，读 Vis-Dial） |
| VCR | 视觉常识推理数据集 | E063 | 原书 p.408：视觉常识推理（口播 1 处，读 V-C-R） |
| GQA | 组合式视觉问答数据集 | E063 | 原书 p.408：建在 Visual Genome 场景图上（口播 1 处，读 G-Q-A） |
| DALL·E | 达利图像生成模型 | E063 | 原书 p.406–407：VQ-VAE-2+Transformer 文生图（口播 3 处，读 DALL-E） |
| ImageNet | 图像网络数据集 | E064 | 原书 p.414：大规模分类数据集，也是预训练与分布漂移题的基准（口播 3 处，读 Image-Net） |
| ImageNet Sketch | 图像网络素描版 | E064 | 原书 p.415：Ex 6.9 的素描分布漂移测试集（口播 1 处，读 Image-Net Sketch） |
| Imagenette | 图像网络小选集 | E064 | 原书 p.414：Ex 6.3 建议的小规模子集（口播 1 处，读英文音） |
| one-hot | 独热编码 | E064 | 原书 p.415：Ex 6.6 的类别向量表示（口播 3 处，读 one hot） |
| Detectron2 | 检测工具箱第二代 | E064 | 原书 p.413：Ex 6.1 作者常用工具箱（口播 2 处，读 Detectron-2） |
| EfficientNet | 高效网络 | E064 | 原书 p.414：Ex 6.3 与 ResNet/ViT 并列的骨干（口播 1 处，读 Efficient-Net） |
| MS COCO | 微软上下文常见物体数据集 | E064 | 原书 p.415：Ex 6.7 训练与比较检测器的数据集（口播 1 处，读 M-S COCO） |
| Faster-RCNN | 更快区域卷积网络（原书连写） | E064 | 原书 p.415：Ex 6.7 原书连写形态（口播 1 处，读 Faster R-C-N-N） |
| PASCAL VOC | 视觉对象类挑战赛 | E064 | 原书 p.412：早期分割挑战（口播 1 处，读 PASCAL V-O-C） |
| Harris | 哈里斯角点检测 | E066 | 原书 p.425：Harris 与 Stephens 1988 行列式-迹测点（口播 5 处，读英文音 Harris） |
| Shi-Tomasi | 史-托马西最小特征值检测 | E066 | 原书 p.425：Shi 与 Tomasi 1994 最小特征值判据（口播 4 处，读英文音） |
| ANMS | 自适应非极大值抑制 | E066 | 原书 p.426–427：局部极大+半径 r 内强 10%（口播 4 处，读字母 A-N-M-S） |
| patch | 图块 | E066 | 原书 p.422：待匹配的局部像素块（口播 2 处，读英文音） |
| Hessian | 海森矩阵 | E066 | 原书 p.426：二阶偏导矩阵/Harris 检测的局部曲率（口播 1 处，读英文音） |
| affine | 仿射 | E067 | 原书 p.431：仿射形变/仿射不变性（口播 4 处，读中文「仿射」，与 affine transformation 同义） |
| MSER | 最大稳定极值区域 | E067 | 原书 p.432：全灰度阈值下面积变化最平的区域（口播 3 处，读字母 M-S-E-R） |
| SURF | 加速鲁棒特征 | E067 | 原书 p.433：积分图加速的检测子（口播 1 处，读英文音 SURF） |
| FAST | 快速角点检测 | E067 | 原书 p.433：早期学习型检测子（口播 2 处，读英文音 FAST） |
| BRISK | 二值鲁棒不变可扩展关键点 | E067 | 原书 p.433：尺度空间 FAST+二值描述子（口播 1 处，读英文音 BRISK） |
| ORB | 定向 FAST 与旋转 BRIEF | E067 | 原书 p.433：给 FAST 加方向（口播 1 处，读字母 O-R-B） |
| KAZE | 非线性扩散尺度空间 | E067 | 原书 p.433：非线性扩散选尺度（口播 1 处，读英文音 KAZE） |
| GLOH | 梯度位置-方向直方图 | E068 | 原书 p.437：对数极坐标分箱、272 维经 PCA 投影到 128 维（口播 7 处，读字母 G-L-O-H） |
| RootSIFT | 开方归一化 SIFT | E068 | 原书 p.436：L1 归一化后逐分量开方提升可区分性（口播 3 处，读英文音 Root-SIFT） |
| L1 | L1 范数 | E068 | 原书 p.436：RootSIFT 的 L1 归一化（口播 1 处，读字母 L-1） |
| NNDR | 最近邻距离比率 | E069 | 原书 p.444：最近邻距离/次近邻距离的比率测试；互检版本更好（口播 5 处，读字母 N-N-D-R） |
| ROC | 受试者工作特征曲线 | E069 | 原书 p.443：真阳率对假阳率的曲线（口播 3 处，读字母 R-O-C） |
| AUC | 曲线下面积 | E069 | 原书 p.443：ROC 曲线下面积，越大越好（口播 3 处，读字母 A-U-C） |
| k-d | k 维树 | E069 | 原书 p.446：沿轴对齐超平面切分的多维搜索树（口播 2 处，读 k-d tree） |
| MOPS | 多尺度定向图块 | E069 | 原书 p.446：四象限求和得三元素哈希索引（口播 1 处，读 MOPS） |
| GPU | 图形处理器 | E069 | 原书 p.445：现代 GPU 可承受全对全比较（口播 1 处，读 G-P-U） |
| FLANN | 快速近似最近邻库 | E069 | 原书 p.447：现代近似最近邻库（口播 1 处，读 FLANN） |
| Faiss | 向量检索库 | E069 | 原书 p.447：现代近似最近邻库（口播 1 处，读 Faiss） |
| SuperGlue | 超级胶水匹配网络 | E069 | 原书 p.448：深度网络匹配过滤系统（口播 1 处，读 Super-Glue） |
| Hough | 霍夫变换 | E070 | 原书 p.478：边缘点投票找直线的变换（口播 10 处，读中文「霍夫」） |
| LSD | 线段检测器 | E070 | 原书 p.481：自底向上分组的开源线段检测器（口播 3 处，读字母 L-S-D） |
| Douglas-Peucker | 道格拉斯-普克折线简化 | E070 | 原书 p.477：递归最远点的折线简化算法（口播 1 处，读中文「道格拉斯-普克」） |
| BRIEF | 二进制独立基本特征 | E071 | 原书 p.438：比较 128 对像素得 128 位向量（口播 2 处，读英文音 BRIEF） |
| FREAK | 快速视网膜关键点 | E071 | 原书 p.439：视网膜 log-polar 采样加比特级联（口播 2 处，读英文音 FREAK） |
| DAISY | 稠密立体匹配描述子 | E071 | 原书 p.438：用立体真值优化参数的稠密描述子（口播 1 处，读英文音 DAISY） |
| Hamming | 汉明距离 | E071 | 原书 p.438：二进制描述子按位比较的快速距离（口播 2 处，读英文音 Hamming） |
| SuperPoint | 整图特征点网络 | E071 | 原书 p.439：整图输入、检测与描述一体的深度特征（口播 2 处，读英文音 SuperPoint） |
| DELF | 深度局部特征 | E071 | 原书 p.439：整图输入的深度局部特征（口播 1 处，读字母 D-E-L-F） |
| D2-Net | 深度检测描述网络 | E071 | 原书 p.439：整图输入的检测描述网络（口播 3 处，读 D-2-Net） |
| ContextDesc | 上下文描述子 | E071 | 原书 p.439：利用上下文信息的整图描述子（口播 1 处，读 Context-Desc） |
| R2D2 | 可靠可重复检测描述子 | E071 | 原书 p.439：整图输入的可靠可重复特征（口播 2 处，读 R-2-D-2） |
| ASLFeat | 自适应尺度局部特征 | E071 | 原书 p.439：整图输入的自适应尺度特征（口播 1 处，读 A-S-L-Feat） |
| CAPS | 通用局部特征 | E071 | 原书 p.439：整图输入的通用局部特征（口播 2 处，读英文音 CAPS） |
| LIFT | 学习型不变特征变换 | E071 | 原书 p.439：patch-based 深度描述子（口播 1 处，读英文音 LIFT） |
| TFeat | 学习型特征 | E071 | 原书 p.439：patch-based 深度描述子（口播 1 处，读 T-Feat） |
| L2-Net | L2 归一化网络 | E071 | 原书 p.439：patch-based 深度描述子（口播 2 处，读 L-2-Net） |
| HardNet | 难样本网络 | E071 | 原书 p.439：patch-based 深度描述子（口播 4 处，读 Hard-Net） |
| Geodesc | 几何描述子 | E071 | 原书 p.439：patch-based 深度描述子（口播 1 处，读 Geo-desc） |
| LF-Net | 学习型局部特征网络 | E071 | 原书 p.439：patch-based 深度描述子（口播 1 处，读 L-F-Net） |
| SOSNet | 二阶相似网络 | E071 | 原书 p.439：patch-based 深度描述子（口播 1 处，读 S-O-S-Net） |
| Key.Net | 关键点网络 | E071 | 原书 p.439：patch-based 深度描述子（口播 3 处，读 Key-Dot-Net） |
| S2DNet | 稀疏到稠密网络 | E071 | 原书 p.440：自称当时最好的描述子之一（口播 1 处，读 S-2-D-Net） |
| LISRD | 局部不变稀疏可靠描述子 | E071 | 原书 p.440：自称当时最好的描述子之一（口播 1 处，读字母 L-I-S-R-D） |
| WISW | 宽基线图像匹配基准 | E071 | 原书 p.440：认为加强版传统 SIFT 最好的基准（口播 1 处，读字母 W-I-S-W） |
| HPatches | 图像块匹配基准 | E071 | 原书 p.440：评图像块匹配的描述子基准（口播 2 处，读 H-Patches） |
| full-image | 整图 | E071 | 原书 p.439：整图输入、感受野可学的描述子路线（口播 2 处，读中文「整图」） |
| patch-based | 基于图像块 | E071 | 原书 p.439：以图像块为输入的深度描述子路线（口播 2 处，读中文「基于图像块」） |
| log-polar | 对数极坐标 | E071 | 原书 p.439：FREAK 的视网膜式采样（口播 2 处，读英文音 log-polar） |
| OpenCV | 开源计算机视觉库 | E071 | 原书无此词；下一集实验课工具名（口播 1 处，读字母 Open-C-V） |
| KNN | K 近邻 | E072 | 原书 p.441–443：最近邻匹配留 k 个候选；本集 k=2（口播 4 处，读字母 K-N-N） |
| crossCheck | 交叉检查 | E072 | 原书无此词；OpenCV BFMatcher 的互检开关（口播 2 处，读中文「交叉检查」） |
| benchmark | 基准测试 | E072 | 原书 p.448：特征性能对比的统一测试口径（口播 2 处，读中文「基准测试」） |
| CPU | 中央处理器 | E072 | 原书无此词；单次耗时受 CPU 调度影响的说明（口播 1 处，读字母 C-P-U） |
| AR | 增强现实 | E073 | 原书无此词；复盘举例「地标检索、商品识别、AR 定位」（口播 1 处，读字母 A-R） |
| LK | 卢卡斯-卡纳德（简称） | E075 | Lucas-Kanade 的简称（口播 8 处，读字母 L-K） |
| LM | 列文伯格-马夸特（简称） | E075 | Levenberg–Marquardt 的简称（口播 1 处，读字母 L-M） |
| DLT | 直接线性变换 | E076 | Direct Linear Transform，单应初值的线性解法（口播 12 处，读字母 D-L-T） |
| sin | 正弦 | E078 | 三角函数，原书式 8.45/8.50 的坐标分量（口播读中文） |
| cos | 余弦 | E078 | 三角函数，同上 |
| MPEG | 运动图像专家组标准 | E082 | Moving Picture Experts Group，视频压缩标准（口播读字母） |
| HEVC | 高效视频编码 | E082 | High Efficiency Video Coding，H.265 标准（口播读字母） |
| NCC | 归一化互相关 | E083 | Normalized Cross-Correlation，减均值/除标准差 |
| SAD | 绝对差之和 | E083 | Sum of Absolute Differences，L1 范数 |
| RMS | 均方根误差 | E083 | Root Mean Square，重叠面积归一化后开方 |
| SSD (sum of squared differences) | 平方差之和 | E083 | 运动估计的 SSD；与 E061 检测器 SSD 同名不同义 |
| CT | 计算机断层扫描 | E085 | 医学配准里的 computed tomography；本集口播「计算机断层 CT」读 C-T |
| MRI | 磁共振成像 | E085 | 医学配准里的 magnetic resonance imaging；本集口播「磁共振 MRI」读 M-R-I |
| Cholesky | 乔列斯基 | E085 | 裸词读音条目；完整名见 Cholesky factorization（E004）。本集口播「Cholesky 分解也就是乔列斯基分解」 |
| KLT | 卡纳德-卢卡斯-托马西特征跟踪 | E085 | Kanade–Lucas–Tomasi 跟踪；E086 展开，本集片尾移交首提 |
| IMU | 惯性测量单元 | E088 | 手机/相机内置加速度计+陀螺仪，给短时运动先验；防抖工程常与视觉估计合用 |
| inpaint | 图像修补 | E088 | 用周围像素补边界缺失/遮挡空洞；本集口播「inpaint」 |
| softmax | 归一化指数函数 | E088 | 把一组分数变成非负、和为 1 的权重；softmax splatting 用它合并多帧贡献 |
| visibility | 可见性 | E088 | 该像素在中间时刻是否被遮挡；插帧判可见性再决定用谁的颜色 |
| L2 | L2 范数（平方惩罚） | E088 | 与 L1 对照的二次惩罚；L2-Net 是另一条（E071） |
| Horn–Schunck | 霍恩-舒克 | E089 | 全局平滑光流：加平方平滑约束同时解所有流向量（p.580）；口播「Horn–Schunck」 |
| PSF | 点扩散函数 | E091 | Point Spread Function；镜头、抗锯齿与传感器面积造成的总模糊 |
| ISP | 图像信号处理器 | E091 | Image Signal Processor；把传感器原始数据加工成可用图像的管线 |
| Bayer | 拜耳阵列 | E091 | 彩色滤镜阵列；每像素只采一个颜色，需去马赛克 |
| gamma | 伽马 | E091 | 显示/编码的非线性亮度映射；口播「gamma」 |
| BM3D | 三维块匹配去噪 | E091 | Block-Matching and 3D filtering，经典图像去噪算法 |
| bracket | 包围曝光 | E091 | 同一场景拍多张不同曝光，供 HDR 合并 |
| JPEG | 联合图像专家组格式 | E092 | 有损压缩图像格式；ISP 处理后经 DCT/量化/压缩得到 |
| ISO | 感光度 | E092 | 相机模拟增益设置；越高越亮、噪声也越大 |
| RAW | 原始数据格式 | E092 | 未经过 ISP 成片的传感器数据；可近似线性 |
| ICC | 国际色彩联盟配置文件 | E092 | International Color Consortium profile；描述相机/显示器色彩响应 |
| YCbCr | 亮度-色度编码 | E092 | JPEG 常用的颜色空间；Y 亮度、Cb/Cr 色度 |
| log2 | 以二为底的对数 | E092 | 曝光值 EV 的定义用 log2 亮度比 |
| AEB | 自动包围曝光 | E093 | Auto Exposure Bracketing；同一机位连拍多张不同曝光，供 HDR 合并 |
| EXIF | 可交换图像文件格式 | E093 | 相机写入照片的元数据，含曝光时间/光圈/ISO 等 |
| base | 基底层 | E094 | 局部色调映射中低通后的 log 亮度大范围层 |
| detail | 细节层 | E094 | log 亮度减去基底层得到的高频层 |
| WLS | 加权最小二乘 | E094 | Weighted Least Squares；保边平滑/局部色调映射的二次型优化 |
| dodging / burning | 局部加减光 | E094 | 暗房局部加光/减光，局部色调映射的直观前身 |
| OpenEXR | 开放 EXR 格式 | E094 | ILM 的 HDR 格式，每通道 16 位浮点 |
| PFM | 便携浮点图 | E094 | Portable FloatMap；PPM 的浮点扩展 |
| PPM | 便携像素图 | E094 | Portable PixMap；简单无压缩图像格式 |
| TIFF | 标签图像文件格式 | E094 | 支持浮点 HDR 值的通用图像格式 |
| JPEG XR | JPEG 扩展范围格式 | E094 | 支持 HDR 的 JPEG 扩展标准 |
| luminance | 亮度 | E094 | 颜色通道的加权亮度，色调映射常只压这一路 |
| low-pass | 低通 | E094 | 保留大范围变化、抹掉细节的滤波 |
| PNG | 便携网络图形 | E094 | 常见 8/16 位显示图像格式，不存 HDR 辐亮度 |
| dodging | 局部减光 | E094 | 暗房局部压暗，对应局部色调映射少压暗部 |
| burning | 局部加光 | E094 | 暗房局部提亮，对应局部色调映射多提暗部 |
| RGB | 红绿蓝三通道 | E094 | 加色三原色通道；HDR 常直接存浮点 RGB |
| dB | 分贝 | E095 | 对数功率/幅度比；PSNR 常用 dB 表示 |
| IBP | 迭代反投影 | E095 | Iterative Back Projection；把重投影误差反投回高分辨图 |
| GMRF | 高斯马尔可夫随机场 | E095 | Gaussian Markov Random Field；二次导数先验对应的图像模型 |
| Tikhonov | 吉洪诺夫正则 | E095 | 二次正则化，用平滑先验稳定病态反问题 |
| NLM | 非局部均值 | E096 | Non-Local Means；用相似块加权平均去噪 |
| TV | 全变分 | E096 | Total Variation；分段平滑先验，保边但会 staircase |
| DnCNN | 去噪残差卷积网络 | E096 | 学习噪声残差的深度去噪基线 |
| SIDD | 智能手机图像去噪数据集 | E096 | Smartphone Image Denoising Dataset；真实噪声基准 |
| Wiener | 维纳滤波 | E096 | 频域最小均方反卷积，需要噪声/核模型 |
| BM3D-Net | BM3D 网络化 | E096 | 把 BM3D 展开/学习化的深度变体 |
| CBDNet | 盲去噪网络 | E096 | 同时估计噪声水平并去噪的深度网络 |
| FFDNet | 快速灵活去噪网络 | E096 | 输入噪声水平图的深度去噪网络 |
| EPLL | 期望块对数似然 | E096 | Expected Patch Log Likelihood；块先验去噪 |
| K-SVD | K 次奇异值分解字典学习 | E096 | 稀疏过完备字典学习算法 |
| TNRD | 可训练非线性反应扩散 | E096 | Trainable Nonlinear Reaction Diffusion |
| WNNM | 加权核范数最小化 | E096 | Weighted Nuclear Norm Minimization；低秩先验去噪 |
| staircase | 阶梯效应 | E096 | 全变分去噪在平滑区产生的阶梯状伪影 |
| deblurring | 去模糊 | E096 | 估计并去除模糊核造成的退化 |
| omega | 去雾强度系数 | E097 | 暗通道透射率估计中保留少量雾的系数，常取 0.95 |
| t0 | 透射率下限 | E097 | 恢复时 max(t,t0) 的下限，常取 0.1，防止除零/噪声爆炸 |
| beta | 大气散射系数 | E097 | 透射率 t=exp(−beta·d) 中的散射系数 |
| Laplacian | 拉普拉斯矩阵 | E097 | 软抠图中用于平滑项的 matting Laplacian 矩阵 |
| alpha | 透明度/抠图 alpha | E097 | 软抠图里与透射率同构的 alpha 图 |
| inpainting | 图像修补 | E098 | 补上被移除物体/缺陷留下的洞 |
| onion-peel | 洋葱剥皮顺序 | E098 | 从洞边界一圈圈向内的填充顺序 |
| partial convolution | 部分卷积 | E098 | 只对已知像素做卷积、掩码随层传播的修补网络 |
| Telea | Telea 快速行进修补 | E098 | 用快速行进法从边界向内填充的经典修补算法 |
| Criminisi | Criminisi 示例块修补 | E098 | 按置信度×数据项排序、从源区搬块的示例块修补 |
| motion | 运动 | E099 | 帧间/物体运动，配准与鬼影处理的输入 |
| metering | 测光 | E099 | 运动测光按场景运动决定帧数与曝光 |
| Night Sight | 谷歌夜视模式 | E099 | 手机低光多帧对齐/合并/运动测光的代表系统 |
| trimap | 三分图 | E100 | 把图像分成前景/背景/未知三区的抠图辅助图 |
| track | 特征轨迹 | E101 | 同一个三维点在多张图里的像素链 |
| COLMAP | COLMAP 重建系统 | E101 | 增量式 SFM/MVS 的开源代表系统 |
| PnP | 透视 n 点位姿 | E101 | 已知 3D 点与 2D 像素求相机位姿 |
| ORB-SLAM | ORB-SLAM 系统 | E101 | 基于 ORB 特征的视觉 SLAM 代表系统 |
| cheirality | 手性/点在相机前方 | E102 | 选 R/t 符号的约束：三维点须在两个相机前方 |
| DEGENSAC | DEGENSAC 退化检测 | E102 | 检测共面点并更可靠估基础矩阵的 RANSAC 变体 |
| RQ | RQ 分解 | E104 | 把相机矩阵前 3×3 拆成上三角 K 与旋转 R（P=K[R\|t]，原书 p.694） |
| P3P | 透视三点问题 | E104 | perspective-3-point-problem；3 对 3D-2D 对应求位姿，第 4 点消歧（原书 p.693 脚注） |
| bas-relief | 浅浮雕歧义 | E104 | 最小 PnP 的深度翻转/浅浮雕歧义（原书 p.695） |
| soft-L1 | 软 L1 鲁棒损失 | E104 | 外点鲁棒核，残差大时近似线性增长（原书 p.695 鲁棒核 ρ 的一种，讲者实现） |
| backprop | 反向传播（简称） | E104 | backpropagation 的简称；链式求导同套路（讲者白话） |
| BA | 束调整（简称） | E105 | bundle adjustment 的简称；联合优化相机与三维点的重投影误差 |
| bundle | 光线束 | E105 | 从相机中心射向三维点的一束光线；bundle adjustment 的词源 |
| adjustment | 调整 | E105 | 对重投影误差的迭代最小化；bundle adjustment 的词源 |
| Schur | 舒尔补 | E105 | 消去点块后得到的相机简化系统 S=Hcc−Hpc^T Hpp^-1 Hpc（原书 p.721） |
| rig | 相机组/支架 | E105 | 多个相机刚性固定在一起的整体；共享外参可一起标定（原书 p.719） |
| Huber | 胡贝尔损失 | E105 | 阈值内二次、阈值外线性的鲁棒核；降低外点话语权 |
| Gauss-Newton | 高斯-牛顿法 | E105 | 用雅可比构造法方程、迭代求解非线性最小二乘的框架 |
| Levenberg-Marquardt | 列文伯格-马夸特（连字符写法） | E105 | 与表内 Levenberg–Marquardt（en dash）同一条；阻尼自适应 GN/LM |
| VSDF | 变状态维滤波 | E105 | variable state dimension filter；只保留窗口内相机/点，控制 BA 规模 |
| Ceres | Ceres 求解器 | E105 | 开源非线性最小二乘/BA 库（原书 p.722） |
| OpenSfM | OpenSfM 系统 | E105 | 开源 SFM/BA 工具（原书 p.722） |
| MVS | 多视图立体 | E106 | Multi-View Stereo；稀疏 SFM 之后做稠密表面的下一步 |
| GPS | 全球定位系统 | E106 | 长序列/户外重建可作弱先验，抑制漂移 |
| depth | 深度 | E106 | 口播借用英文词，指三维点在相机坐标系下的 Z 值 |
| OpenMVG | OpenMVG 系统 | E107 | 开源全局/增量 SFM 管线，含旋转平均与中心求解 |
| Theia | Theia 系统 | E107 | 开源 SFM 管线，含全局旋转/中心技术 |
| ATE | 绝对轨迹误差 | E108 | Absolute Trajectory Error；估计轨迹与真值的均方根位置误差 |
| PTAM | 并行跟踪与建图 | E108 | Parallel Tracking and Mapping；前后端分线程的经典 SLAM |
| DSO | 直接稀疏里程计 | E108 | Direct Sparse Odometry；只在强梯度处保留深度 |
| DTAM | 稠密跟踪与建图 | E108 | Dense Tracking and Mapping；逐帧稠密深度 |
| VIO | 视觉惯性里程计 | E108 | Visual-Inertial Odometry；IMU+视觉互补 |
| BAD SLAM | 束调整直接 RGB-D SLAM | E108 | Bundle Adjusted Direct RGB-D SLAM |
| LSD-SLAM | 大尺度直接 SLAM | E108 | Large-Scale Direct SLAM；半稠密直接法 |
| MonoSLAM | 单目 SLAM | E108 | 早期单目视觉 SLAM 系统 |
| RGB-D | 彩色+深度 | E108 | 同时含颜色与深度的传感器/数据 |
| SVO | 半直接视觉里程计 | E108 | Semi-direct Visual Odometry；块跟踪+BA |
| iSAM | 增量平滑与建图 | E108 | Incremental Smoothing and Mapping；因子图增量优化 |
| BAD | BAD SLAM 的简称 | E108 | Bundle Adjusted Direct RGB-D SLAM 的简称 |
| Essential Graph | 本质图 | E109 | ORB-SLAM 的稀疏位姿图：生成树+高共视边+回环边，保留强约束网络 |
| Essential | 本质图（简称） | E109 | Essential Graph 的简称；口播按英文词面读 |
| Graph | 图 | E109 | 节点+边的结构；此处指共视图/本质图 |
| DBoW2 | 词袋库 DBoW2 | E109 | 离线视觉词表+倒排索引的地点识别库；用于回环检测与重定位 |
| motion-only | 仅运动优化 | E109 | 只优化当前相机位姿、不动地图点的 BA；跟踪线程每帧使用 |
| PINHOLE | 针孔相机模型 | E110 | 只用焦距和主点、无畸变的相机模型；COLMAP 相机参数 700,700,400,300 |
| SINGLE | 单相机模式 | E110 | COLMAP 的相机分组模式：所有图像共用同一内参 |
| pycolmap | COLMAP 的 Python 接口 | E110 | 调用 COLMAP 核心的 Python 绑定；本集用它跑特征/匹配/增量重建 |
| SQLite | 嵌入式数据库 | E110 | COLMAP 的 database.sqlite 存储特征、匹配和相机 |
| sparse | 稀疏重建 | E110 | 只保留特征点的相机位姿+三维点模型；与稠密 MVS 相对 |
| SGM | 半全局匹配 | E111 | Semi-Global Matching；多条一维路径近似全局平滑，工程常用的立体匹配算法 |
| ToF | 飞行时间 | E111 | Time-of-Flight；通过光往返时间测距的深度传感器 |
| WTA | 赢家通吃 | E112 | Winner-Take-All；每个像素选聚合代价最小的视差 |
| DSI | 视差空间图 | E112 | Disparity Space Image；x-y-d 三维代价体 |
| census | 普查变换 | E112 | 把窗口邻居与中心比较成位向量，抗曝光变化 |
| DP | 动态规划 | E113 | Dynamic Programming；在代价矩阵上找最小代价路径 |
| MRF | 马尔可夫随机场 | E113 | Markov Random Field；全局立体匹配常用的能量模型 |
| SIMD | 单指令多数据 | E113 | Single Instruction Multiple Data；实时 SGM 常用向量化加速 |
| FPGA | 现场可编程门阵列 | E113 | Field-Programmable Gate Array；低功耗实时 SGM 的常见载体 |
| PSM-Net | 金字塔立体匹配网络 | E113 | Pyramid Stereo Matching Network；E114 主题 |
| SPP (spatial pyramid pooling) | 空间金字塔池化 | E114 | 多尺度平均池化后上采样拼接，扩大感受野 |
| Middlebury | Middlebury 立体评测 | E114 | 高分辨率立体匹配公开基准，训练数据少 |
| DispNet-C | 视差估计网络 DispNet-C | E114 | 端到端视差网络，使用相关层计算左右相似度 |
| KITTI | KITTI 数据集 | E114 | 自动驾驶立体/光流公开数据集 |
| SGM-Net | SGM 惩罚网络 | E114 | 用 CNN 预测 SGM 转移代价/惩罚项 |
| Depth Anything | Depth Anything 单目深度模型 | E115 | 大规模无标注数据预训练的单目深度基础模型 |
| MiDaS | MiDaS 单目深度网络 | E115 | 多数据集联合训练的相对深度网络 |
| NYU | NYU 室内深度数据集 | E115 | NYU Depth Dataset V2，室内场景 |
| NYUv2 | NYUv2 室内深度数据集 | E115 | NYU Depth Dataset V2 的简称 |
| Cityscapes | Cityscapes 街景数据集 | E115 | 城市街景语义/深度数据集 |
| DINOv2 | DINOv2 自监督视觉模型 | E115 | 提供语义先验的预训练编码器 |
| LiDAR | 激光雷达 | E115 | 主动测距传感器，可提供度量深度真值 |
| Make3D | Make3D 数据集 | E115 | 早期单目深度数据集 |
| SOTA | 当前最优 | E115 | state of the art |
| ScanNet | ScanNet 室内三维数据集 | E115 | 室内场景 RGB-D 与三维重建数据集 |
| Kinect | Kinect 深度相机 | E116 | Microsoft 低成本 RGB-D 相机 |
| MEMS | 微机电系统 | E116 | 微型振镜/扫描器件 |
| V2 | 第二代 | E116 | Kinect V2 的版本号 |
| Stanford Cart | 斯坦福推车 | E117 | Moravec 早期视觉导航小车；原书 p.738 |
| CMU | 卡内基梅隆大学 | E117 | Carnegie Mellon University；其早期自动驾驶项目最早用神经网络驾驶 |
| DARPA | 美国国防高级研究计划局 | E117 | Defense Advanced Research Projects Agency；Grand/Urban Challenge |
| Google | 谷歌 | E117 | 原书 p.738 产业研究例子 |
| Tesla | 特斯拉 | E117 | 原书 p.738 产业研究例子 |
| Skydio | Skydio 无人机公司 | E117 | 原书 p.739 自主飞行无人机例子 |
| VaMoRs | VaMoRs 自动驾驶系统 | E117 | Dickmanns 团队 25Hz/100km/h 系统 |
| bad-1 | 1 像素坏点率 | E118 | 误差超过 1 像素的像素比例 |
| bad-2 | 2 像素坏点率 | E118 | 误差超过 2 像素的像素比例 |
| Table | 表 | E118 | 原书 Table 12.1 的表号 |
| stereo | 立体 | E118 | stereo matching 的简短词面；全称条已收 |
| DTU | DTU 数据集 | E118 | 124 个玩具场景的多视图数据集 |
| ETH3D | ETH3D 数据集 | E118 | 13 训练 + 12 测试的高分辨率场景 |
| Freiburg Scene Flow | 弗赖堡场景流数据集 | E118 | 39k 合成立体对 |
| closed-world | 封闭世界 | E118 | 训练/测试同分布的数据设定 |
| Marching Cubes | 行进立方体 | E119 | 把隐式场等值面提取成三角网格；E120 展开 |
| Poisson | 泊松 | E118 | Poisson reconstruction 的短词面 |
| TSDF | 截断符号距离场 | E120 | 深度图融合常用的体表示 |
| DeepSDF | 深度符号距离场网络 | E120 | 用网络学习符号距离 |
| KinectFusion | Kinect 融合系统 | E120 | 深度图实时融合与相机跟踪 |
| Occupancy Networks | 占据网络 | E120 | 用网络学习 0/1 占据 |
| MC | 行进立方体（简称） | E120 | Marching Cubes 的缩写 |
| Marching Squares | 行进方块（二维等值线） | E120 | Marching Cubes 的二维版本 |
| NeRF | 神经辐射场 | E121 | NeRF 短词面；全称条已收 |
| ICP | 迭代最近点 | E120 | ICP 短词面；全称条已收 |
| range data merging | 范围数据合并 | E121 | 多帧深度/距离图对齐后合并 |
| VRIP | 体范围图像处理 | E121 | Volumetric Range Image Processing |
| Octomap | 八叉树占据地图 | E121 | 八叉树+概率占据 |
| Voxel Hashing | 体素哈希 | E121 | 空间哈希压缩 TSDF |
| Chisel | Chisel 实时重建系统 | E121 | 空间哈希压缩 TSDF 的系统 |
| point-to-point | 点到点 | E121 | ICP 的对应点误差 |
| point-to-plane | 点到面 | E122 | ICP 的对应面误差（E121 预告） |
| raycast | 光线投射 | E121 | 从 TSDF 渲染或提面 |
| sdf | 符号距离函数 | E121 | signed distance function 的缩写 |
| RMSE | 均方根误差 | E122 | 对应点距离的均方根 |
| KD-tree | KD 树 | E122 | 最近邻空间索引 |
| Procrustes | 普氏分析 | E122 | 正交 Procrustes 求旋转 |
| splash | splash 表示 | E122 | 比自旋图像更早的局部描述子 |
| x' | x 撇 | E122 | 目标点坐标 |
| c' | c 撇 | E122 | 目标点云质心 |
| Chen | 陈（作者） | E122 | Chen and Medioni 1992 |
| Horn | 霍恩（作者） | E122 | 绝对定向四元数方法 |
| Medioni | 梅迪奥尼（作者） | E122 | Chen and Medioni 1992 |
| MSE | 均方误差 | E123 | mean squared error；与 RMSE 区分 |
| SIREN | 正弦激活网络 | E123 | sinusoidal representation networks |
| Adam | 亚当优化器 | E123 | adaptive moment estimation |
| Tancik | 坦尼克（作者） | E123 | Tancik et al. 2020 位置编码分析 |
| IDR | 隐式可微渲染器 | E123 | Implicit Differentiable Renderer |
| NSVF | 神经稀疏体素场 | E123 | Neural Sparse Voxel Fields |
| pixelNeRF | 像素级 NeRF | E123 | 单图/少图条件化 NeRF |
| inside-out | 由内向外 | E123 | 1/r 反转球参数化 |
| xyz | 三维坐标 | E123 | 空间位置坐标的简写 |
| 3DGS | 三维高斯泼溅 | E124 | 3D Gaussian Splatting；DGS 为扫描器切分 |
| Barron | 巴伦（作者） | E124 | 锥形追踪/积分位置编码 |
| HyperNeRF | 超神经辐射场 | E124 | 动态 NeRF 变体 |
| MPI | 多平面图像 | E124 | multiplane images |
| NeRD | 神经反射分解 | E124 | Neural Reflection Decomposition |
| NeRF-W | 野外神经辐射场 | E124 | NeRF in the Wild |
| NeRV | 神经反射与可见性场 | E124 | Neural Reflectance and Visibility Fields |
| PhySG | 物理着色高斯 | E124 | SDF + 球面高斯 BRDF |
| nerfies | 可变形神经辐射场 | E124 | Deformable NeRF |
| Space-time | 时空 | E124 | Space-time NeRF 的简写 |
| batch | 批 | E124 | 训练批/射线批 |
| Neural Scene Flow Fields | 神经场景流场 | E124 | 动态 NeRF 变体 |
| Dynamic NeRF | 动态神经辐射场 | E124 | 动态 NeRF 变体 |
| tile | 瓦片/分块 | E125 | 屏幕分块光栅化单位 |
| D-SSIM | 结构差异度量 | E125 | 1−SSIM，论文损失项 |
| Mip-NeRF360 | Mip-NeRF360 | E125 | 高质量慢速 NeRF 基线 |
| CUDA | 英伟达并行计算平台 | E125 | GPU 通用计算接口 |
| Kerbl | 克布尔（作者） | E125 | 3DGS 第一作者 |
| LPIPS | 学习感知图像块相似度 | E125 | 感知指标 |
| SGD | 随机梯度下降 | E125 | stochastic gradient descent |
| lambda | 权重系数 λ | E125 | 损失权重 |
| overdraw | 过度绘制 | E125 | 同一像素被多次覆盖 |
| popping | 跳变/突现伪影 | E125 | 大高斯导致的视角跳变 |
| sigmoid | S 形函数 | E125 | 把不透明度压到 0–1 |
| surfels | 面元 | E125 | surface elements 的复数 |
| M-NeRF360 | Mip-NeRF360 方法（论文基线） | E126 | 3DGS 论文 Table 1 的 NeRF 基线 |
| Ours-30K | 本文方法（30K 迭代配置） | E126 | 3DGS 论文 Table 1 的 3DGS 配置 |
| Tanks&Temples | Tanks and Temples 数据集 | E126 | 论文评测数据集 |
| A6000 | 英伟达 A6000 GPU | E126 | 论文训练硬件 |
| FLIP | 图像差异评估器 | E126 | 教材 p.148 提到的感知指标 |
| FPS | 每秒帧数 | E126 | frames per second |
| Imax | 最大信号值 | E126 | PSNR 公式中的峰值 |
| log10 | 以 10 为底的对数 | E126 | PSNR 计算 |
| per-scene | 逐场景 | E126 | 每场景平均/最差 |
| MB | 兆字节 | E126 | 显存单位 |
| Tanks and Temples | Tanks and Temples 数据集 | E127 | 多视图/重建评测数据集 |
| Cramér–Rao | 克拉美–罗下界 | E127 | 参数估计不确定度下界 |
| ICL-NUIM | ICL-NUIM 室内数据集 | E127 | 合成室内 RGB-D 数据集 |
| NYU3D | NYU3D 室内数据集 | E127 | 早期室内三维数据集 |
| PASCAL | PASCAL 数据集 | E127 | 经典视觉数据集 |
| 3D Scenes | 3D Scenes 数据集 | E127 | 室内场景数据集 |
| ShapeNet | ShapeNet 数据集 | E127 | 三维物体数据集 |
| Sintel | Sintel 数据集 | E127 | 光流评测数据集 |
| tanh | 双曲正切 | E128 | 坐标 MLP 的激活函数 |
| numpy | NumPy 数值库 | E128 | Python 数值计算库 |
| warp | 图像变形/重投影 | E129 | 按几何把像素搬到新位置 |
| blend | 混合 | E129 | 按权重叠加多幅图像 |
| view morphing | 视角变形 | E129 | 视图插值+morphing，处理非刚性 |
| EPI | 极线平面图 | E130 | epipolar plane image；沿相机位置切 EPI，轨迹斜率编码深度 |
| Lumigraph | 光场图（带深度补偿） | E130 | 光场变体，用表面三维位置补偿重采样 |
| Lumisphere | 方向光球 | E130 | 表面光场为每个表面点存所有可见方向的出射光 |
| marker | 标记点 | E130 | 合成实验中用于测轨迹/锐度的彩色标记 |
| Lytro | Lytro 光场相机 | E130 | 消费级光场相机代表；本集只说原理，不实拍 |
| ULR | 无结构光场图 | E130 | unstructured Lumigraph，按五条保真准则直接选相机 |
| Photo Tourism | 照片漫游 | E130 | 互联网照片三维漫游系统；正课在 E131 |
| Photosynth | Photosynth 照片漫游系统 | E131 | Microsoft 的照片漫游系统 |
| Photo Tours | 照片漫游（Google Maps 功能） | E131 | 自动铺虚拟路径 |
| sea of images | 照片海/图像海 | E131 | 密集图像集合，支持连续移动 |
| virtual path | 虚拟路径 | E131 | 邻近照片连成的路径 |
| Aspen MovieMap | Aspen 电影地图 | E131 | 早期交互式街景系统 |
| Ladybug | Ladybug 全景相机 | E131 | 六路相机拼 360° |
| Street View | 街景 | E131 | Google 街景 |
| 360cities | 360cities 全景社区 | E131 | 360° 视频虚拟旅行网站 |
| orbit | 环绕视角 | E131 | 绕物体一圈的视角 |
| orbits | 环绕视角（复数） | E131 | orbit 的复数 |
| walkthrough | 漫游/导览 | E131 | 交互式视频漫游 |
| viewpoint | 视点 | E131 | 观察位置 |
| image-based rendering (IBR) | 基于图像的绘制 | E131 | IBR 全称；E132 展开 |
| map correlation | 地图相关/地图对齐 | E131 | 视频序列与地图对齐 |
| Google Maps | 谷歌地图 | E131 | Photo Tours/Street View 平台 |
| AI | 人工智能 | E132 | 生成模型与神经渲染的上下文 |
| VR | 虚拟现实 | E132 | E133 展开空间计算 |
| neural rendering | 神经渲染 | E132 | 生成式机器学习+图形学物理知识 |
| neural codes | 神经编码 | E132 | 把源图编码成可重投影的特征 |
| neural texture | 神经纹理 | E132 | Deferred Neural Rendering 的二维神经纹理图 |
| deep blending | 深度混合 | E132 | 学习每个新视角的逐像素混合权重 |
| billboard | 广告牌平面/定向平面 | E132 | GeLaTO 用少量定向平面表示薄透明物体 |
| ray marching | 光线步进 | E132 | 体素/体积渲染沿射线采样 |
| soft splat | 软泼溅 | E132 | SynSin 按深度软权重把特征投到屏幕 |
| texture map | 纹理图 | E132 | 二维神经纹理图，渲染时采样解码 |
| over compositing | over 合成 | E132 | 从后往前的 alpha 合成 |
| novel view synthesis | 新视角合成 | E132 | 六类神经渲染任务之一 |
| free-viewpoint video | 自由视点视频 | E132 | 六类神经渲染任务之一 |
| relighting | 重打光 | E132 | 六类神经渲染任务之一 |
| facial reenactment | 人脸重演 | E132 | 六类神经渲染任务之一 |
| body reenactment | 人体重演 | E132 | 六类神经渲染任务之一 |
| semantic photo synthesis | 语义照片合成 | E132 | 六类神经渲染任务之一 |
| appearance vector | 外观向量 | E132 | Meshry 为每幅图+深度图算的潜在外观 |
| latent appearance | 潜在外观 | E132 | 可调外观的低维表示 |
| IBRNet | IBRNet 邻图神经渲染 | E132 | 从邻近输入图采样神经特征再做体积合成 |
| LookinGood | LookinGood 人体渲染系统 | E132 | U-Net 补洞/去噪/提分辨率 |
| Free View Synthesis | 自由视点合成系统 | E132 | 神经编码+重投影+循环网络+softmax |
| Deferred Neural Rendering | 延迟神经渲染 | E132 | 学二维神经纹理，渲染时采样解码 |
| Meshry | Meshry 外观可控渲染 | E132 | 潜在外观向量+深度图 |
| GeLaTO | GeLaTO 生成潜纹理物体 | E132 | 少量定向平面 billboard+神经纹理 |
| SynSin | SynSin 单图新视角 | E132 | 单张彩色图→神经特征+深度 |
| Semantic View Synthesis | 语义视图合成 | E132 | 语义标签图→彩色图+深度→MPI |
| DeepVoxels | DeepVoxels 体素神经渲染 | E132 | 学习三维神经编码 |
| Neural Volumes | Neural Volumes 体积渲染 | E132 | 多视图→三维 RGBα 体积 |
| AdaIN | 自适应实例归一化 | E132 | 用外观向量调制编码器-解码器 |
| coordinate MLP | 坐标多层感知器 | E132 | 隐函数的全连接网络实现 |
| implicit function | 隐函数 | E132 | 连续函数表示场景 |
| volumetric rendering | 体积渲染 | E132 | 射线采样+体积合成 |
| RGBα | RGBα 体积 | E132 | 颜色+不透明度的体积表示 |
| virtual viewpoint video | 虚拟视点视频 | E133 | 自由视点视频的另一种叫法 |
| scene flow | 场景流 | E133 | 对应表面元的三维运动 |
| 6DoF | 六自由度 | E133 | 六自由度视频，允许有界体积内移动 |
| DoF | 自由度 | E133 | degrees of freedom 的缩写 |
| ODS | 全向立体视频 | E133 | omnidirectional stereo；Jump 输出格式 |
| Virtualized Reality | 虚拟现实重建系统 | E133 | 早期多相机 3D 视频系统 |
| Immersive Video | 沉浸式视频系统 | E133 | 早期多视图视频系统 |
| Image-Based Visual Hulls | 基于图像的视觉外壳 | E133 | 用视觉外壳做新视角渲染 |
| Jump | Jump 全景立体捕捉系统 | E133 | Google 16 台 GoPro 环拍系统 |
| GoPro | GoPro 运动相机 | E133 | Jump 使用的相机 |
| Zitnick | Zitnick 虚拟视点视频系统 | E133 | 两层深度图+边界抠图 |
| Vedula | Vedula 多视图视频研究 | E133 | scene flow 与时空插值 |
| omnidirectional stereo | 全向立体 | E133 | ODS 全称 |
| sprite | 图像贴片（精灵） | E134 | 带深度的图像小面片，按深度做 warp 绘制；原书 p.886 与 LDI 并列为图像式表示变体 |
| LDI | 分层深度图 | E134 | layered depth image；每像素存多层颜色+深度，可绕开一致三维几何（原书 p.886） |
| ARC3D | ARC3D 网页建模系统 | E134 | 完整网页端全自动 3D 建模系统（原书 p.854） |
| Pix4D | Pix4D 摄影测量软件 | E134 | 商业摄影测量系统，物体和场景都能拍（原书 p.855） |
| VideoTrace | VideoTrace 交互建模系统 | E134 | 自动跟踪、恢复结构，让用户在点云上画三角面；人机协作（原书 p.855） |
| radial basis function (RBF) | 径向基函数 | E135 | 固定中心+输出权的小型学习模型；本集 16 中心，不是神经网络 |
| Nayar | Nayar 第一性原理讲座 | E135 | Shree Nayar《First Principles of Computer Vision》；原书 p.917 引用 |
| 3D photography | 三维摄影 | E134 | 补录 E134 漏收：多图建模型+恢复表面外观（原书 p.854）；E136 复盘中作总称 |
| image-based modeling | 基于图像的建模 | E136 | 从一张或多张图得到部分/完整三维模型（原书 p.29） |
| shape from X | 由 X 恢复形状 | E136 | 阴影/纹理/焦距等经典形状线索（原书 p.29） |
| Python | Python 语言 | E137 | 原书 p.962：作者近年从 C++ 转向 Python；交互环境+NumPy |
| SciPy | SciPy 科学计算库 | E137 | 本集计时用 scipy.ndimage；不是 OpenCV 替代品 |
| Colab | Google Colab | E137 | 原书 p.964：免费 GPU/Jupyter，适合课程作业 |
| Kaggle | Kaggle 竞赛平台 | E137 | 原书 p.964：在流行数据集上比拼模型 |
| Open3D | Open3D 三维库 | E137 | 三维重建航线推荐工具（课程整理，不是教材原文推荐） |
| Snavely | Snavely 视觉课程 | E137 | 原书 p.970：Cornell 视觉导论课 |
| DiLiGenT | DiLiGenT 光度立体数据集 | E137 | 原书 p.960：校准光照+真值形状 |
| VisionBib | VisionBib 数据集索引 | E137 | 原书 p.954：Keith Price 维护的旧数据集列表 |
| Java | Java 语言 | E137 | 原书 p.962：OpenCV 接口之一 |
| MATLAB | MATLAB 语言 | E137 | 原书 p.962：OpenCV 接口之一 |
| Jupyter | Jupyter 笔记本 | E137 | 原书 p.964：Colab 的交互环境 |
| Papers with Code | Papers with Code 论文代码索引 | E137 | 原书 p.961：论文+实现+榜单入口 |
| 1DSfM | 1DSfM 地标图像集 | E137 | 原书 p.961：Photo Tourism 数据集的扩展 |
| snake | 活动轮廓 | E138 | 原书 p.466：能量最小化二维样条；单数口播，复数见 E062「snakes」 |
| Canny | Canny 边缘检测 | E138 | 原书 p.457：非极大值抑制+双阈值滞后；John Canny 1986 |
| k-means | k 均值聚类 | E138 | 原书 p.487：作为参数化聚类对照；本集 lab 基线，不是 mean shift |
| mean shift | 均值漂移 | E138 | 原书 p.487：非参数密度峰值搜索 |
| F1 | F1 分数 | E138 | 精度与召回的调和平均；本集 lab 指标 |
| Lab | Lab 颜色空间 | E138 | 本集 lab 用 OpenCV Lab；原书 p.487 例子用 L*u*v*，不是同一空间 |
| NP-complete | NP 完全 | E138 | 原书 p.490：最优归一化割 NP-complete |
| Dijkstra | Dijkstra 算法 | E138 | 原书 p.473：智能剪刀最低成本路径 |
| findContours | 轮廓提取函数 | E138 | OpenCV 函数；本集 lab 基线，不声称书上代码 |
| rotoscoping | 逐帧抠像/旋转遮罩 | E138 | 原书 p.476：跟踪轮廓变形手绘动画 |
| Felzenszwalb | Felzenszwalb | E138 | 原书 p.486：图分割合并算法作者 |
| Huttenlocher | Huttenlocher | E138 | 原书 p.486：图分割合并算法作者 |
| ICM | 迭代条件模式 | E139 | 原书 p.216：逐点翻转降能量，易卡局部最小 |
| BP | 信念传播 | E139 | 原书 p.219：树上精确、带环近似 |
| N4 | 四邻域 | E139 | 原书 p.214：上下左右四个邻居 |
| N8 | 八邻域 | E139 | 原书 p.215：含对角线八个邻居 |
| TIN | 三角不规则网络 | E139 | 原书 p.194：Delaunay 三角网 |
| Delaunay | Delaunay 三角化 | E139 | 原书 p.194：避免狭长三角形 |
| alpha-expansion | α 扩展 | E139 | 原书 p.219：多标签图割近似 |
| alpha-beta swap | α-β 交换 | E139 | 原书 p.219：多标签图割近似 |
| ill-posed | 病态（解不唯一） | E139 | 原书 p.204：多解/对输入敏感 |
| ill-conditioned | 病态（数值不稳） | E139 | 原书 p.204：小变化引起大变化 |
| inverse problem | 反问题 | E139 | 原书 p.204：从数据反推函数 |
| strokes | 交互笔画 | E139 | 原书 p.222：用户画前景/背景笔画 |
| Gibbs | Gibbs 分布 | E139 | 原书 p.214：MRF 的 Boltzmann/Gibbs 形式 |
| Boltzmann | Boltzmann 分布 | E139 | 原书 p.214：同上 |
| GraphCut | 图割 | E139 | 原书 p.216：二值 MRF 全局最小 |
| GrabCut | GrabCut | E139 | 交互分割经典实现；书上未点名，讲者补充 |
| DeepLab | DeepLab | E139 | 分割网络/CRF 后处理；书上未点名，讲者补充 |
| dense CRF | 全连接条件随机场 | E139 | 原书 p.225：全连接/均值场 |
| loopy belief propagation | 带环信念传播 | E139 | 原书 p.219：环图上近似 |
| Hardy multiquadric | Hardy 多二次 | E139 | 原书 p.196：RBF 核之一 |
| Terzopoulos | Terzopoulos | E139 | 原书 p.206：受控连续样条 |
| crease | 折痕 | E139 | 原书 p.207：受控连续的局部折痕 |
| tear | 撕裂 | E139 | 原书 p.207：受控连续的局部断裂 |
| C0 continuity | C0 连续 | E139 | 原书 p.195：分片线性插值 |
| C1 continuity | C1 连续 | E139 | 原书 p.195：一阶导连续 |
| NP-hard | NP 难 | E139 | 原书 p.219：一般多标签 MRF |
| k1 | 径向畸变一阶系数 | E140 | 原书 p.691 式 11.10：x(1+κ1r²+κ2r⁴) |
| k2 | 径向畸变二阶系数 | E140 | 原书 p.691 式 11.10：同上 |
| Zhang | 张正友平面标定 | E140 | 原书 p.686：对 B 的九个元素构造线性约束 |
| IAC | 绝对二次曲线的像 | E140 | 原书 p.687：B=K⁻ᵀK⁻¹ |
| N-planes | N 平面标定 | E140 | 原书 p.686：平面靶在空间移动 |
| fx | 像素焦距 x | E140 | 内参矩阵 K 的 [0,0] 元 |
| fy | 像素焦距 y | E140 | 内参矩阵 K 的 [1,1] 元 |
| matte | 遮罩/蒙版 | E141 | 原书 p.892：每层含颜色图、matte、合成顺序 |
| mattes | 遮罩（复数） | E141 | matte 的复数；原书 p.892 |
| HMM | 隐马尔可夫模型 | E141 | 原书 p.891：转移概率编码 HMM |
| face | 人脸 | E141 | face transfer 的词面 |
| transfer | 迁移 | E141 | face transfer 的词面 |
| face transfer | 面部迁移 | E141 | 原书 p.888：新源视频驱动旧视频 |
| Video Rewrite | 视频重写系统 | E141 | 原书 p.888：重排旧视频帧匹配新语音 |
| Vid2Player | 网球视频重合成系统 | E141 | 原书 p.891：建模网球运动员动作与镜头 |
| loop | 循环 | E141 | 视频纹理的循环播放 |
| mocap | 动作捕捉 | E141 | 原书 p.891：mocap soup 的素材来源 |
| environment | 环境 | E141 | environment matte 的词面 |
| environment matte | 环境蒙版 | E141 | 下集 §14.4：折射/反射环境的蒙版 |
| video texture | 视频纹理 | E141 | 原书 p.889：短视频重排成无限长动画 |
| motion texture | 运动纹理 | E141 | 原书 p.892：每层的随机位移场 |
| video-based rendering | 基于视频的渲染 | E141 | 原书 p.887：从拍到的视频生成新序列 |
| video-based animation | 基于视频的动画 | E141 | 原书 p.887：重排/修改视频素材 |
| jump table | 跳转表 | E141 | 原书 p.891：任意两帧间的转移概率 |
| LSF | 线扩散函数 | E142 | 原书 p.617：斜边法恢复的一维投影 |
| Richardson–Lucy deconvolution | Richardson–Lucy 去卷积 | E142 | 本集下游示意：用估计 PSF 迭代去模糊 |
| Kruppa equations | Kruppa 方程 | E143 | 原书 p.713：从 F 的两视图约束解焦距 |
| self-calibration | 自标定 | E143 | 原书 p.712：仅用对应点把投影重建升为度量重建 |
| trifocal tensor | 三焦张量 | E143 | 原书 p.684 明确跳过推导；本集仅讲边界与线传递 |
