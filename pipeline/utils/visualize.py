import os
import numpy as np
import matplotlib.pyplot as plt


def draw_results(imgs1, imgs2, labels, pred_labels, savepath, n=4):   
    fig, ax = plt.subplots(n, 2, figsize=(10, 5))
    for idx, (img1, img2, gt_label, pr_label) in enumerate(zip(imgs1, imgs2, labels, pred_labels)):
        if idx < n:
            ax[idx,0].imshow(img1.detach().float().cpu().numpy().transpose(1, 2, 0))
            ax[idx,0].axis("off")

            ax[idx,1].imshow(img2.detach().float().cpu().numpy().transpose(1, 2, 0))
            ax[idx,1].axis("off")
            
            gt_label = gt_label.detach().float().cpu().numpy()
            pr_label = np.round(pr_label.detach().float().cpu().numpy())

            ax[idx,0].set_title(f'GT_LABEL: {gt_label}')
            ax[idx,1].set_title(f'PRED_LABEL: {pr_label}')
        else:
            break
    
    plt.savefig(os.path.join(savepath, 'results.jpg'))
    plt.close()