# Tribology-Classification-of-wear-in-human-joints
Classification of Progressive Wear on a Multi-Directional Pin-on-Disc Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo Using Acoustic Emission and Machine Learning

# Journal link

[10.3390/lubricants12020047](https://www.mdpi.com/2075-4442/12/2/47)

# Overview


Human joint prostheses face wear and related failure mechanisms due to the complex tribological contact between Ultra-High-Molecular-Weight Polyethylene (UHMWPE) and Cobalt-Chromium-Molybdenum (CoCrMo). This study investigates wear mechanisms with the long term goal to predict failure rates in human joint prostheses. Multi-directional pin-on-disc tests were conducted in a medium simulating real in-vivo conditions to analyze the wear behavior of UHMWPE pins sliding against CoCrMo discs. The objectives were to gain insights into wear mechanisms and to classify wear rates. Real-time wear monitoring was enabled using Acoustic Emission (AE) sensors, capturing signals before and after weekly visual inspections over 2.3 million cycles. This approach facilitated continuous wear data collection and wear progression assessment. Integrating AE sensors proved valuable in detecting wear-related signals, enhancing wear progression detection, and aiding in anticipation of failure. This study presents a novel approach for monitoring wear progression in human joint prostheses using two Machine Learning (ML) frameworks. The first framework involved manually extracting time, frequency, and time-frequency domain features from acoustic signatures based on human knowledge. ML classifiers, including Logistic Regression, Support Vector Machine, k-Nearest Neighbor, Random Forest, Neural Networks, and Extreme Gradient Boosting, were applied for wear classification, achieving an average accuracy between 81% to 89%. The second framework introduced a contrastive learning-based Convolutional Neural Network (CNN) with circle loss to enhance wear classification performance. CNN extracted feature maps from the acoustic signatures, which were then used to retrain the ML classifiers. This approach demonstrated superior classification performance of 94% to 96% compared to manual feature extraction. Machine learning techniques enabled accurate wear classification and improved progressive assessment of human joint prostheses. Automated feature extraction using the contrastive learning-based CNN provided better insights into wear patterns and enhanced the predictive capabilities of ML classifiers. This approach can improve the early detection of wear-related failures and enable timely interventions. The successful implementation of AE sensors for real-time monitoring in lab simulated conditions demonstrates their effectiveness in detecting wear-related signals and supporting proactive measures to prevent wear failures. This unique method of monitoring and predicting wear progression using AE and ML in the UHMWPE-CoCrMo pairing enhances understanding of wear mechanisms in UHMWPE. This knowledge can guide the development of more reliable and durable prosthetic joint designs.  

# Experimental setup


![1](https://github.com/vigneashpandiyan/Tribology-Classification-of-wear-in-human-joints/assets/39007209/e931ab6f-9b72-4240-b0de-90efb47d619a)
![2](https://github.com/vigneashpandiyan/Tribology-Classification-of-wear-in-human-joints/assets/39007209/97f32a33-4a31-4fd9-ae40-eb745dab1d74)

# Methodology

![image](https://github.com/vigneashpandiyan/Tribology-Classification-of-wear-in-human-joints/assets/39007209/b36043e5-92f6-435c-87a7-c8af5d4f2a79)

# Results

![www_screencapture_com_2024-2-8_12_49-ezgif com-video-to-gif-converter](https://github.com/vigneashpandiyan/Tribology-Classification-of-wear-in-human-joints/assets/39007209/feb80de9-0df2-421d-b308-fb595d945c2c)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Tribology-Classification-of-wear-in-human-joints

cd  Tribology-Classification-of-wear-in-human-joints
python  ..../Manual feature extraction/Main Features.py
python  ..../Contrastive learner [Circle loss]/Main_Circle.py
python ..../Classification/Main Classification.py
python ..../Classification/Contrastive Classification.py
```

# Citation
```

@Article{lubricants12020047,
AUTHOR = {Deshpande, Pushkar and Wasmer, Kilian and Imwinkelried, Thomas and Heuberger, Roman and Dreyer, Michael and Weisse, Bernhard and Crockett, Rowena and Pandiyan, Vigneashwara},
TITLE = {Classification of Progressive Wear on a Multi-Directional Pin-on-Disc Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo Using Acoustic Emission and Machine Learning},
JOURNAL = {Lubricants},
VOLUME = {12},
YEAR = {2024},
NUMBER = {2},
ARTICLE-NUMBER = {47},
URL = {https://www.mdpi.com/2075-4442/12/2/47},
ISSN = {2075-4442},
ABSTRACT = {Human joint prostheses experience wear failure due to the complex interactions between Ultra-High-Molecular-Weight Polyethylene (UHMWPE) and Cobalt-Chromium-Molybdenum (CoCrMo). This study uses the wear classification to investigate the gradual and progressive abrasive wear mechanisms in UHMWPE. Pin-on-disc tests were conducted under simulated in vivo conditions, monitoring wear using Acoustic Emission (AE). Two Machine Learning (ML) frameworks were employed for wear classification: manual feature extraction with ML classifiers and a contrastive learning-based Convolutional Neural Network (CNN) with ML classifiers. The CNN-based feature extraction approach achieved superior classification performance (94% to 96%) compared to manual feature extraction (81% to 89%). The ML techniques enable accurate wear classification, aiding in understanding surface states and early failure detection. Real-time monitoring using AE sensors shows promise for interventions and improving prosthetic joint design.},
DOI = {10.3390/lubricants12020047}
}

```
