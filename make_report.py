"""Build a complete Udacity ML Capstone PDF from saved, verified experiment artifacts.

Usage: python make_report.py --results runs --out capstone_report.pdf
Requires both test_metrics.json and aligned test_predictions.csv files; never fabricates results.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, PageBreak, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable, KeepTogether)
from analyze_results import calculate, plot_results

NAVY=colors.HexColor('#152E46'); BLUE=colors.HexColor('#265B80'); INK=colors.HexColor('#233748')
MUTED=colors.HexColor('#617383'); LIGHT=colors.HexColor('#EEF3F7'); BORDER=colors.HexColor('#D5E0E8')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='C_title',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=27,leading=34,textColor=NAVY,spaceAfter=24,alignment=TA_LEFT))
styles.add(ParagraphStyle(name='C_subtitle',parent=styles['Normal'],fontName='Helvetica',fontSize=15,leading=22,textColor=BLUE,spaceAfter=16))
styles.add(ParagraphStyle(name='C_h',parent=styles['Heading1'],fontName='Helvetica-Bold',fontSize=16.5,leading=23,textColor=NAVY,spaceAfter=12))
styles.add(ParagraphStyle(name='C_h2',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11.6,leading=16.4,textColor=BLUE,spaceBefore=8,spaceAfter=7))
styles.add(ParagraphStyle(name='C_body',parent=styles['BodyText'],fontName='Helvetica',fontSize=9.55,leading=14.8,textColor=INK,spaceAfter=10))
styles.add(ParagraphStyle(name='C_small',parent=styles['BodyText'],fontName='Helvetica',fontSize=8.8,leading=13,textColor=INK,spaceAfter=7))
styles.add(ParagraphStyle(name='C_caption',parent=styles['Normal'],fontName='Helvetica',fontSize=8,leading=11,textColor=MUTED,spaceBefore=4,spaceAfter=12,alignment=TA_CENTER))
styles.add(ParagraphStyle(name='C_table',parent=styles['Normal'],fontName='Helvetica',fontSize=8.6,leading=12,textColor=INK))
styles.add(ParagraphStyle(name='C_table_header',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=8.7,leading=12,textColor=colors.white))
styles.add(ParagraphStyle(name='C_kicker',parent=styles['Normal'],fontName='Helvetica-Bold',fontSize=9,leading=13,textColor=BLUE,spaceAfter=9))
styles.add(ParagraphStyle(name='C_quote',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=10.7,leading=16,textColor=NAVY,spaceAfter=14))

# Content is written as basic ReportLab paragraph markup, with occasional bold keywords.
def P(x, style='C_body'): return Paragraph(x,styles[style])
def add(flow,*paras,style='C_body'):
    for x in paras: flow.append(P(x,style))
def h(flow,title):flow.append(P(title,'C_h2'))
def fig(flow,path,width=440,maxheight=250,caption=''):
    if not Path(path).exists(): raise FileNotFoundError(f'Required report figure missing: {path}')
    from PIL import Image as PILImage
    with PILImage.open(path) as im: w,hgt=im.size
    scale=min(width/w,maxheight/hgt)
    flow.append(Image(str(path),width=w*scale,height=hgt*scale))
    if caption:flow.append(P(caption,'C_caption'))

def tbl(flow,rows,widths=None,header=True):
    data=[[P(str(v),'C_table_header' if i==0 and header else 'C_table') for v in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1 if header else 0)
    cmd=[('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),9),
         ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
         ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,-1),(-1,-1),0.6,BORDER),
         ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT])]
    if header:cmd.append(('BACKGROUND',(0,0),(-1,0),NAVY))
    t.setStyle(TableStyle(cmd));flow.append(t);flow.append(Spacer(1,12))

def page(fl,number,title,section):
    if fl:fl.append(PageBreak())
    fl.append(P(section.upper(),'C_kicker'))
    fl.append(P(f'{number}. {title}','C_h'))

def footer(c,doc):
    c.saveState();w,h=doc.pagesize
    if doc.page==1:
        c.setFillColor(NAVY);c.rect(0,h-14,w,14,fill=1,stroke=0)
    else:
        c.setStrokeColor(BORDER);c.setLineWidth(.7);c.line(54,52,w-54,52)
        c.setFillColor(MUTED);c.setFont('Helvetica',8)
        c.drawString(54,37,'MACHINE LEARNING CAPSTONE  |  CIFAR-10 / RESNET18')
        c.drawRightString(w-54,37,str(doc.page))
    c.restoreState()

def readj(root,sub):
    fp=root/sub
    if not fp.exists():raise FileNotFoundError(f'Missing {fp}; the report cannot invent experiment measurements.')
    return json.loads(fp.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='runs');ap.add_argument('--out',default='capstone_report.pdf');args=ap.parse_args()
    r=Path(args.results)
    f=readj(r,'frozen/test_metrics.json');t=readj(r,'fine_tune/test_metrics.json')
    fm=readj(r,'frozen/training_meta.json');tm=readj(r,'fine_tune/training_meta.json')
    ds=readj(r,'dataset_info.json')
    summary=calculate(r)
    # Generate the two comparison figures and JSON in the same way as analyze_results.py.
    (r/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_results(r,summary)
    rows={name:list(csv.DictReader((r/name/'history.csv').open(newline=''))) for name in ('frozen','fine_tune')}
    if f['test_count']!=450 or t['test_count']!=450 or summary['test_count']!=450:
        raise ValueError('Unexpected test size: amend the report narrative before regenerating.')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    doc=BaseDocTemplate(str(out),pagesize=(612,792),leftMargin=54,rightMargin=54,
                        topMargin=61,bottomMargin=67,title='Animal Image Classification using Transfer Learning with ResNet18',
                        author='Udacity Machine Learning Capstone Project')
    frame=Frame(54,67,504,664,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='standard',frames=[frame],onPage=footer))
    fl=[]
    # PAGE 1 - Cover
    fl.append(Spacer(1,75));fl.append(P('UDACITY  /  MACHINE LEARNING CAPSTONE','C_kicker'))
    fl.append(P('Animal Image Classification<br/>Using Transfer Learning<br/>with ResNet18','C_title'))
    fl.append(P('A controlled comparison of frozen-feature learning and partial fine-tuning','C_subtitle'))
    fl.append(HRFlowable(width='100%',thickness=1,color=BORDER,spaceBefore=10,spaceAfter=28))
    add(fl,'<b>Research question.</b> Does unfreezing the final residual block of an ImageNet-pretrained ResNet18 improve animal-image classification compared with training a replacement classifier head alone?')
    add(fl,'<b>Dataset.</b> Balanced CIFAR-10 subset of 2,700 cat, dog and horse images: 1,800 training, 450 validation and 450 held-out testing examples.')
    add(fl,'<b>Summary finding.</b> Fine-tuning improved held-out accuracy from 86.00% to 90.89% and macro-F1 from 0.8590 to 0.9090. The paired accuracy difference was +4.89 percentage points (95% paired bootstrap interval: +1.78 to +8.00 points).')
    fl.append(Spacer(1,31));fl.append(P('EXPERIMENT RECORD  |  SEED 42  |  3 EPOCHS PER MODEL  |  APPLE MPS','C_kicker'))
    # PAGE 2
    page(fl,1,'Project overview and problem definition','Definition')
    add(fl,'<b>Background.</b> Image categorization is a practical computer-vision problem: software must infer the subject of a photograph from pixel patterns, even when pose, lighting and background vary. A useful small-scale example is distinguishing cats, dogs and horses, a task relevant to photo organization and a simplified prototype for visual tagging. The application here is educational rather than a production animal-identification service.')
    add(fl,'<b>Motivation.</b> Training a deep image model from randomly initialized parameters generally requires substantial labeled data and computation. Transfer learning is attractive for smaller datasets because the early and intermediate visual representations can be reused from a model pretrained on a large dataset. ResNet uses residual shortcuts to facilitate optimization of deep convolutional networks [2].')
    add(fl,'<b>Problem statement.</b> Given one 32 x 32 RGB image from a restricted three-category CIFAR-10 subset, predict exactly one label: cat, dog or horse. The network outputs three logits, converts them to class probabilities using softmax at inference, and selects the highest-probability label. This is supervised, single-label multiclass classification.')
    h(fl,'Hypothesis and scope')
    add(fl,'The primary hypothesis is that updating ResNet18 <b>layer4</b> in addition to the final classification layer will improve held-out macro-F1 compared with updating the classification layer alone. Both strategies begin from ImageNet-1K pretrained weights, use the same data manifest and are selected by validation macro-F1. The smaller frozen-feature model is the learned benchmark.')
    add(fl,'The aim is not to estimate performance for all photographs, every animal category or deployment environments. It is to provide a measurable, reproducible experiment with a clear baseline, refinement, held-out validation and transparent limitations.')
    # PAGE 3
    page(fl,2,'Metrics, benchmarks and experimental controls','Definition')
    add(fl,'<b>Primary metric: macro-F1.</b> For each class, precision = TP/(TP + FP), recall = TP/(TP + FN), and F1 = 2 x precision x recall/(precision + recall). Macro-F1 is the arithmetic mean of the three classwise F1 scores. It gives cat, dog and horse equal importance, even when one class is harder to recognize.')
    add(fl,'<b>Complementary measures.</b> Accuracy is the proportion of examples correctly labeled. Cross-entropy loss is used for training and to observe convergence. Confusion matrices describe which true categories are mistaken for others. We additionally report 95% percentile bootstrap intervals for accuracy and a paired bootstrap interval for the accuracy difference.')
    tbl(fl,[['Benchmark / model','Decision rule','Role'],['Non-learned reference','Always predict a single category','33.33% accuracy on this balanced test set'],['Experiment A','Frozen ResNet18 backbone; train 3-class head','Learned baseline'],['Experiment B','Train head and last residual block (layer4)','Refinement candidate']],widths=[130,174,200])
    h(fl,'Separation of selection and evaluation')
    add(fl,'Training and validation images originate only from the official CIFAR-10 training partition. The held-out images originate only from the official CIFAR-10 test partition. Each model checkpoint is selected using its best <b>validation macro-F1</b>; test-set measurements are calculated after this selection. The saved manifest preserves the exact sampled indices and class labels.')
    add(fl,'<b>Predeclared success criterion.</b> The refined strategy should improve test macro-F1 over the learned baseline, ideally with a paired accuracy interval excluding zero. This combines practical effect size with uncertainty assessment. A 33.33% class-constant baseline is a mathematical reference, not a third trained experiment.')
    # PAGE 4
    page(fl,3,'Dataset and class exploration','Analysis')
    add(fl,'CIFAR-10 contains 60,000 RGB images of size 32 x 32 across ten categories, with 50,000 images in the official training partition and 10,000 in the official test partition [1, 5]. We select the original labels cat=3, dog=5 and horse=7. A seeded permutation picks distinct indices within each selected category, retaining the source-partition boundary.')
    tbl(fl,[['Split','Cat','Dog','Horse','Total'],['Training','600','600','600','1,800'],['Validation','150','150','150','450'],['Test','150','150','150','450'],['Total','900','900','900','2,700']],widths=[132,90,90,90,102])
    fig(fl,r/'figures/class_distribution.png',width=460,maxheight=255,caption='Figure 1. Every class has equal representation within each data split; the plotted counts are computed from the saved split manifest.')
    add(fl,'<b>Data integrity.</b> The split manifest contains an original-source identifier and integer row index for every example. The included unit test verifies the requested counts, uniqueness of source/index pairs and exclusive use of the official test partition for test examples. This prevents index reuse across local splits; it does not guarantee all photographs are semantically independent.')
    # PAGE 5
    page(fl,4,'Exploratory visualization and data challenges','Analysis')
    fig(fl,r/'figures/sample_images.png',width=460,maxheight=360,caption='Figure 2. First four sampled training examples per class; originally 32 x 32 pixels, enlarged for viewing.')
    add(fl,'<b>Observed visual characteristics.</b> The dog row includes both a dark full-body outdoor silhouette and a tightly framed face; shape, background and image scale vary considerably even within one label. The cat examples include a dark cat against a light background and a pale cat beside a bright yellow graphic, illustrating substantial color and scene variation. Horse examples range from a close side view to an image with a rider, showing that context is not uniform.')
    add(fl,'<b>Challenges.</b> Enlargement makes pixelation obvious: fine textures, eyes and body boundaries are poorly resolved in many examples. Backgrounds and other contextual cues may be spuriously predictive. Cropping, overlapping animals, pose and label ambiguity could all affect predictions. Resizing to 224 x 224 makes inputs compatible with pretrained ResNet18, but cannot recover information absent in the original 32 x 32 pixels.')
    # PAGE 6
    page(fl,5,'Algorithms, transfer learning and benchmark','Analysis')
    add(fl,'<b>Convolutional residual network.</b> ResNet18 is an 18-layer residual CNN; its shortcut connections let blocks learn residual updates relative to their inputs, improving optimization compared with a plain stack of layers [2]. ImageNet pretraining supplies visual features learned across a much broader object-recognition problem. Following established transfer-learning practice [3,4], the original 1,000-class fully connected layer is replaced with a randomly initialized 3-class linear layer.')
    tbl(fl,[['Setting','Experiment A: frozen','Experiment B: partial tune'],['Initialization','ImageNet1K_V1','ImageNet1K_V1'],['Trainable components','New fully connected head','Head + final residual block'],['Trainable parameters',f"{fm['trainable_parameters']:,}",f"{tm['trainable_parameters']:,}"],['Learning rate',f"{fm['lr']:.4f}",f"{tm['lr']:.4f}"],['Optimizer','AdamW','AdamW'],['Weight decay','0.0001','0.0001']],widths=[162,171,171])
    h(fl,'Why compare these two strategies?')
    add(fl,'The frozen-feature baseline is a computationally cheap, strong alternative to learning every CNN weight from scratch: only 1,539 classifier weights/biases are updated. Allowing the last residual block to adapt gives the network substantially more trainable capacity (8,395,267 parameters), potentially improving adaptation to small CIFAR-10 animal images but increasing susceptibility to overfitting.')
    add(fl,'<b>Important design caveat.</b> Experiment B also uses a lower learning rate (0.0001 versus 0.001). Therefore this is a comparison of two complete training configurations, not a causal isolation of layer4 trainability alone. A future ablation should hold learning rate constant or compare a matched rate grid on validation data.')
    # PAGE 7
    page(fl,6,'Preprocessing and leakage prevention','Methodology')
    add(fl,'<b>Split construction.</b> For each target class, NumPy seeded permutation (seed=42) selects 600 official-train rows for training, the next 150 official-train rows for validation and 150 rows from the official-test partition. Records are written to <font name="Courier">runs/splits.csv</font>, including split, source, original index, target and category name. The same file drives both experiments.')
    tbl(fl,[['Stage','Applied to training','Applied to validation/test'],['Resize','224 x 224 pixels','224 x 224 pixels'],['Horizontal flip','Random, p = 0.5','Not applied'],['Tensor conversion','PIL RGB -> float tensor','Same'],['Channel normalization','ImageNet mean / std','Same'],['Shuffling','Training loader only','Disabled']],widths=[150,177,177])
    add(fl,'The ImageNet normalization mean is (0.485, 0.456, 0.406) and standard deviation is (0.229, 0.224, 0.225), consistent with the pretrained model configuration [4]. Random horizontal flipping only alters training samples, not validation or test observations. No target-label information is introduced through the transforms.')
    h(fl,'Potential data and model leakage controls')
    add(fl,'The model weights are freshly loaded from the same original ImageNet checkpoint for each run; Experiment B does not begin from Experiment A\'s trained three-class checkpoint. The scripts load test examples only for evaluation. Checkpoint choice uses validation macro-F1, and the analysis CSVs are aligned by ordered test position and true label before any paired comparison. These controls support fair evaluation on this fixed subset.')
    # PAGE 8
    page(fl,7,'Implementation and reproducibility','Methodology')
    add(fl,'The pipeline is implemented in <font name="Courier">capstone.py</font> with three CLI stages: <b>prepare</b> downloads/source-checks CIFAR-10 and saves the manifest and EDA figures; <b>train</b> constructs model variants, runs training, records learning curves and saves the best validation checkpoint; <b>evaluate</b> reloads that checkpoint and writes test metrics, aligned predictions and a confusion matrix. A separate analysis script recomputes paired statistics directly from prediction CSVs.')
    tbl(fl,[['Recorded configuration','Value'],['Execution backend','Apple Metal Performance Shaders (mps)'],['PyTorch / Torchvision','2.15.0 / 0.29.2 (reported runtime)'],['Seed / batch size','42 / 32'],['Training epochs per model','3'],['Loss / optimizer','Multiclass cross-entropy / AdamW'],['Best checkpoint rule','Maximum validation macro-F1']],widths=[220,284])
    h(fl,'Implementation complications and resolution')
    add(fl,'During setup, the initially selected environment lacked PyTorch; this was resolved by activating the project virtual environment and installing dependencies. The CIFAR-10 download then failed SSL certificate verification under macOS Python. Installing <font name="Courier">certifi</font> and setting <font name="Courier">SSL_CERT_FILE</font> to its certificate bundle allowed the official dataset to download with certificate verification still enabled. Pretrained weights were then downloaded successfully.')
    add(fl,'All training and test evaluation were run locally; no SageMaker endpoint or ongoing AWS compute was needed. For reproduction, use the README commands, the deterministic manifest, the specified pretrained weight variant, and the saved metrics. Exact numeric replication may still vary slightly across hardware and library versions, especially GPU/MPS backends.')
    # PAGE 9
    page(fl,8,'Refinement and learning-curve behavior','Methodology')
    fig(fl,r/'frozen/learning_curves.png',width=480,maxheight=210,caption='Figure 3. Frozen-feature head: training vs validation accuracy and loss over three epochs.')
    fig(fl,r/'fine_tune/learning_curves.png',width=480,maxheight=210,caption='Figure 4. Fine-tuned layer4 + head: training vs validation accuracy and loss over three epochs.')
    add(fl,'<b>Baseline convergence.</b> Frozen-head validation accuracy grew from 79.11% to 84.00% and validation loss decreased from 0.5414 to 0.4171. Fine-tuning raised best validation accuracy to 88.89% and best validation macro-F1 to 0.8888, versus 0.8398 for the frozen model; both best checkpoints occurred at epoch 3.')
    add(fl,'<b>Overfitting caution.</b> In fine-tuning, training accuracy increased to 97.78% while validation accuracy stayed at 88.89%. Validation loss improved through epoch 2 (0.2835) but rose to 0.2926 at epoch 3, despite a small F1 improvement. This divergence suggests early overfitting/calibration concerns; it does not prove performance would worsen with further training.')
    # PAGE 10
    page(fl,9,'Held-out results and principal findings','Results')
    fig(fl,r/'model_comparison.png',width=450,maxheight=240,caption='Figure 5. Held-out test accuracy and macro-F1 measured on the same 450 selected images.')
    tbl(fl,[['Statistic','Frozen head','Fine-tuned layer4'],['Correct / tested','387 / 450','409 / 450'],['Accuracy','86.00%','90.89%'],['Macro-F1','0.8590','0.9090'],['Accuracy 95% bootstrap CI','82.67%-89.11%','88.22%-93.56%'],['Validation macro-F1 (selected)','0.8398','0.8888']],widths=[218,143,143])
    add(fl,'The fine-tuned model improved accuracy by <b>4.89 percentage points</b> and macro-F1 by <b>0.0501</b> on the fixed held-out subset. Its 409 correct predictions represent 22 additional correct labels relative to the frozen head. The test error count fell from 63 to 41, a 34.9% relative reduction in errors on these images.')
    add(fl,'This performance exceeds both the non-learned 33.33% reference and the trained frozen-feature benchmark. Because the test set is balanced across three classes, accuracy is interpretable, but the macro-F1 increase confirms that the gain is not solely a class-prevalence effect.')
    # PAGE 11
    page(fl,10,'Classwise results and confusion patterns','Results')
    tbl(fl,[['Class','Frozen precision / recall / F1','Fine-tuned precision / recall / F1'],
        *[[c,f"{f['class_report'][c]['precision']:.3f} / {f['class_report'][c]['recall']:.3f} / {f['class_report'][c]['f1-score']:.3f}",
               f"{t['class_report'][c]['precision']:.3f} / {t['class_report'][c]['recall']:.3f} / {t['class_report'][c]['f1-score']:.3f}"] for c in ('cat','dog','horse')]],widths=[75,211,218])
    fig(fl,r/'class_recall_comparison.png',width=465,maxheight=230,caption='Figure 6. Test recall by true class. Fine-tuning improves dog and horse recall, while cat recall falls.')
    add(fl,'<b>Observed errors.</b> Baseline dog recall was 113/150 (75.33%), versus 134/150 (89.33%) after fine-tuning. Dog-to-cat errors decreased from 27 to 13; dog-to-horse errors from 10 to 3. Horse recall rose from 135/150 to 141/150. Conversely, cat recall fell from 139/150 to 134/150 and cat-to-dog mistakes increased from 7 to 12. This class-specific tradeoff is hidden by overall accuracy alone.')
    # PAGE 12
    page(fl,11,'Confusion matrices and matched error analysis','Results')
    # Two source confusion matrices, with axis definitions already rendered inside each.
    pair=Table([[Image(str(r/'frozen/confusion_matrix.png'),width=244,height=212),
                 Image(str(r/'fine_tune/confusion_matrix.png'),width=244,height=212)]],colWidths=[252,252])
    pair.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),2)]))
    fl.append(pair);fl.append(P('Figure 7. Frozen-head (left) and fine-tuned (right) test confusion matrices. Rows are true labels; columns are predictions.','C_caption'))
    tbl(fl,[['Same test image','Count','Interpretation'],['Both models correct',str(summary['both_correct']),'Consistent correct predictions'],['Baseline wrong; fine-tune correct',str(summary['frozen_wrong_fine_tune_correct']),'Previously incorrect cases repaired'],['Baseline correct; fine-tune wrong',str(summary['frozen_correct_fine_tune_wrong']),'New mistakes introduced'],['Both models wrong',str(summary['both_wrong']),'Shared failures']],widths=[205,65,234])
    add(fl,'Matched predictions show 37 repaired mistakes but 15 new mistakes, for a net gain of 22. Most repairs occur in the dog class (24 corrected, 3 newly incorrect), while cat predictions exhibit the opposite pattern (5 corrected, 10 newly incorrect). This is why evaluating the models on exactly the same images is more informative than comparing independent summary scores.')
    # PAGE 13
    page(fl,12,'Statistical validation, robustness and interpretation','Results')
    add(fl,'<b>Paired bootstrap.</b> Let each held-out image contribute +1 if only the fine-tuned model is correct, -1 if only the frozen head is correct, or 0 if the two agree in correctness. The mean of these matched differences is 22/450 = <b>+0.0489</b>. We draw 2,000 bootstrap resamples of 450 matched image positions (with replacement; seed 42) and compute the 2.5th and 97.5th percentiles of the resampled mean difference.')
    tbl(fl,[['Evidence','Estimate'],['Observed paired accuracy gain','+4.89 percentage points'],['95% paired bootstrap interval','+1.78 to +8.00 percentage points'],['Discordant outcomes','37 improved, 15 worsened'],['Exact two-sided McNemar test','p = 0.0032']],widths=[275,229])
    add(fl,'Because the paired bootstrap interval excludes zero, the observed accuracy gain is supported by this image-level resampling analysis. An exact McNemar test on the 52 discordant image pairs also yields p = 0.0032, consistent with different matched error rates. These inferences are conditional on this fixed sampling design, fixed model instances and the assumption that held-out images are representative of the intended target population.')
    h(fl,'Validation and robustness boundaries')
    add(fl,'The dataset has only 450 held-out examples and one fixed random subset; three epochs and one training seed per strategy do not measure training-run variability. Individual confidence intervals overlap slightly and should not be used in place of the paired interval. The task is restricted to three CIFAR-10 labels and to heavily upscaled 32 x 32 pixels; no external photos, fairness groups, shifted imaging conditions or model calibration measurements were evaluated.')
    add(fl,'Moreover, the strategies use different learning rates and trainable parameter counts. Accordingly, the improvement supports the <b>combined fine-tuning configuration</b>, not an isolated causal claim about changing only one layer. The test images were used for final reporting and error analysis, not for choosing the saved checkpoints.')
    # PAGE 14
    page(fl,13,'Conclusion, improvements and references','Conclusion')
    add(fl,'<b>Conclusion.</b> This capstone constructed a reproducible three-class animal classifier and compared two transfer-learning strategies. Partial fine-tuning achieved 90.89% test accuracy and 0.9090 macro-F1 versus 86.00% and 0.8590 for the frozen-feature baseline. The matched test accuracy gain was +4.89 percentage points, with a paired bootstrap 95% interval from +1.78 to +8.00 points. Dog recognition improved substantially; cat recall declined slightly, emphasizing the value of classwise inspection.')
    add(fl,'The tested configuration meets the defined objective on the chosen CIFAR-10 subset, but it is not proof of deployment readiness. The low-resolution, balanced dataset, single split/seed, three-epoch schedule and different optimizer learning rates limit generality and causal interpretation. No real-world robustness claim is made.')
    h(fl,'Next steps')
    add(fl,'A stronger follow-up would compare matched learning-rate grids using validation data, repeat each strategy across several independent seeds, add early stopping, inspect individual high-confidence mistakes, evaluate probability calibration and test a separately curated real-world animal-image dataset. None of these additional experiments is presented as completed work.')
    h(fl,'Reproducibility and references')
    add(fl,'The accompanying files include <font name="Courier">capstone.py</font>, <font name="Courier">analyze_results.py</font>, <font name="Courier">make_report.py</font>, <font name="Courier">requirements.txt</font>, tests, a saved split manifest, both training histories, both test-prediction files, metrics JSON, charts and a setup README. Model checkpoints and raw CIFAR-10 files are intentionally not included; the README explains how to obtain data and rerun training.')
    for line in [
      '[1] Krizhevsky, A. (2009). <i>Learning Multiple Layers of Features from Tiny Images.</i> https://www.cs.toronto.edu/~kriz/cifar.html',
      '[2] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). <i>Deep Residual Learning for Image Recognition.</i> CVPR, pp. 770-778. https://arxiv.org/abs/1512.03385',
      '[3] PyTorch. <i>Transfer Learning for Computer Vision Tutorial.</i> https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial',
      '[4] Torchvision. <i>ResNet18 weights and model documentation.</i> https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html',
      '[5] Torchvision. <i>CIFAR10 dataset API.</i> https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html'
    ]: fl.append(P(line,'C_small'))
    doc.build(fl)
    print(f'Created completed report: {out}')

if __name__=='__main__':main()
