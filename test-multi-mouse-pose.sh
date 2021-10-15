#!/bin/bash

# for((i=1; i<7; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse-${i}.yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse-${i}/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse-${i} \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<11; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2019-11-19_${i}.yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-11-19_${i}/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2019-11-19_${i} \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<9; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2019-12-19_${i}.yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-12-19_${i}/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2019-12-19_${i} \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<2; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2019-12-31_${i}.yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-12-31_${i}/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2019-12-31_${i} \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<17; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2020-01-17_$(printf %02d $i).yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-17_$(printf %02d $i)/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --min-pose-heatmap-val 1.0 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2020-01-17_$(printf %02d $i) \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<10; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2020-01-21_$(printf %02d $i).yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-21_$(printf %02d $i)/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --min-pose-heatmap-val 1.0 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2020-01-21_$(printf %02d $i) \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<13; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2020-01-22_$(printf %02d $i).yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-22_$(printf %02d $i)/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --min-pose-heatmap-val 1.0 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2020-01-22_$(printf %02d $i) \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<8; i++))
# do
#     # python -u tools/testmultimouseinference.py \
#     #     --cfg experiments/multimouse/multimouse_2020-01-30_$(printf %02d $i).yaml \
#     #     --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-30_$(printf %02d $i)/best_state.pth \
#     #     --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#     #     --image-dir data/multi-mouse/Dataset \
#     #     --image-list data/multi-mouse-val-set.txt \
#     #     --max-embed-sep-within-instances 0.3 \
#     #     --min-embed-sep-between-instances 0.3 \
#     #     --min-pose-heatmap-val 1.0 \
#     #     --plot-heatmap \
#     #     --image-out-dir temp/multimouse_2020-01-30_$(printf %02d $i) \
#     #     --dist-out-file output-multi-mouse/dist-out.txt
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2020-01-30_$(printf %02d $i).yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-30_$(printf %02d $i)/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2020-01-30_$(printf %02d $i) \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

# for((i=1; i<13; i++))
# do
#     python -u tools/testmultimouseinference.py \
#         --cfg experiments/multimouse/multimouse_2020-02-03_$(printf %02d $i).yaml \
#         --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-02-03_$(printf %02d $i)/best_state.pth \
#         --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
#         --image-dir data/multi-mouse/Dataset \
#         --image-list data/multi-mouse-val-set.txt \
#         --max-embed-sep-within-instances 0.3 \
#         --min-embed-sep-between-instances 0.3 \
#         --plot-heatmap \
#         --image-out-dir temp/multimouse_2020-02-03_$(printf %02d $i) \
#         --dist-out-file output-multi-mouse/dist-out.txt
# done

for((i=1; i<4; i++))
do
    python -u tools/testmultimouseinference.py \
        --cfg experiments/multimouse/multimouse_2020-02-10_$(printf %02d $i).yaml \
        --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-02-10_$(printf %02d $i)/best_state.pth \
        --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
        --image-dir data/multi-mouse/Dataset \
        --image-list data/multi-mouse-val-set.txt \
        --max-embed-sep-within-instances 0.3 \
        --min-embed-sep-between-instances 0.3 \
        --plot-heatmap \
        --image-out-dir temp/multimouse_2020-02-10_$(printf %02d $i) \
        --dist-out-file output-multi-mouse/dist-out.txt
done
