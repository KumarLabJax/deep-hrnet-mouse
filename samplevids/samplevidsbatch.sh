#!/bin/bash
#
#SBATCH --job-name=sample-vids
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --mem=8G
#SBATCH --nice

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

export PATH="/opt/singularity/bin:${PATH}"
if [[ -n "${SLURM_JOB_ID}" ]]
then
    # the script is being run by slurm
    if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]
    then
        if [[ -n "${BATCH_FILE}" ]]
        then
            # here we use the array ID to pull out the right video
            VIDEO_FILE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}"))
            echo "BEGIN PROCESSING: ${VIDEO_FILE}"
            echo "DUMP OF CURRENT ENVIRONMENT:"
            env

            cd "$(dirname "${BATCH_FILE}")"
            mkdir -p 'vids'
            mkdir -p 'frames'

            rclone copy --include "${VIDEO_FILE}" "labdropbox:/KumarLab's shared workspace/VideoData/MDS_Tests" vids

            if [[ -f "vids/${VIDEO_FILE}" ]]
            then
                module load singularity
                singularity exec "/projects/kumar-lab/USERS/sheppk/poseest-env/multi-mouse-pose-2020-02-12.sif" bash -c "python3 sampleframes.py --videos 'vids/${VIDEO_FILE}' --root-dir vids --outdir frames --neighbor-frame-count 5 --mark-frame"
                rm "vids/${VIDEO_FILE}"

                echo "FINISHED PROCESSING: ${VIDEO_FILE}"
            else
                echo "ERROR: could not find video file: ${VIDEO_FILE}" >&2
            fi
        else
            echo "ERROR: the BATCH_FILE environment variable is not defined" >&2
        fi
    else
        echo "ERROR: no SLURM_ARRAY_TASK_ID found" >&2
    fi
else
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ -f "${1}" ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit batch file: ${1}"
        test_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${test_count} videos"

        # Here we perform a self-submit
        sbatch --export=ROOT_DIR="$(dirname "${0}")",BATCH_FILE="${1}" --array="1-${test_count}%24" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process. Eg: ./samplevidsbatch batchfile.txt" >&2
        exit 1
    fi
fi
