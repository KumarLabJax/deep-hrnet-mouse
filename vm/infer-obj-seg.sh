#!/bin/bash
#
#SBATCH --job-name=infer-obj-seg-arr
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:1
#SBATCH --mem=16G
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
            cd "$(dirname "${BATCH_FILE}")"
            if [[ -f "${VIDEO_FILE}" ]]
            then
                echo "DUMP OF CURRENT ENVIRONMENT:"
                env
                echo "BEGIN PROCESSING: ${VIDEO_FILE}"
                H5_OUT_FILE="${VIDEO_FILE%.*}_obj_seg.h5"
                module load singularity
                singularity run --nv "${ROOT_DIR}/obj-seg-2019-07-17.simg" "${VIDEO_FILE}" "${H5_OUT_FILE}"
                echo "FINISHED PROCESSING: ${VIDEO_FILE}"
            else
                echo "ERROR: could not find configuration file: ${VIDEO_FILE}" >&2
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
        sbatch --export=ROOT_DIR="$(dirname "${0}")",BATCH_FILE="${1}" --array="1-${test_count}" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process. Eg: ./infer-obj-seg.sh batchfile.txt" >&2
        exit 1
    fi
fi
