#!/bin/bash
#
#SBATCH --job-name=extract-frames
#
#SBATCH --qos=batch
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

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
        if [[ ( -n "${BATCH_FILE}" ) && ( -n "${OUT_DIR}" ) ]]
        then

            echo "DUMP OF CURRENT ENVIRONMENT:"
            env
            echo "BEGIN PROCESSING: ${BATCH_FILE} => ${OUT_DIR} for row ${SLURM_ARRAY_TASK_ID}"

            module load singularity
            singularity exec "${ROOT_DIR}/multi-mouse-pose-2020-02-12.sif" python3 "${ROOT_DIR}/extractframes.py" \
                --frame-table "${BATCH_FILE}" \
                --frame-table-row "${SLURM_ARRAY_TASK_ID}" \
                --root-dir "$(dirname "${BATCH_FILE}")" \
                --outdir "${OUT_DIR}"

            echo "FINISHED PROCESSING"

        else
            echo "ERROR: the BATCH_FILE or OUT_DIR environment variable is not defined" >&2
        fi
    else
        echo "ERROR: no SLURM_ARRAY_TASK_ID found" >&2
    fi
else
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ ( -f "${1}" ) &&  ( -n "${2}" ) ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit batch file: ${1}"
        test_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${test_count} videos"

        mkdir -p "${2}"

        # Here we perform a self-submit
        sbatch --export=ROOT_DIR="$(dirname "${0}")",BATCH_FILE="${1}",OUT_DIR="${2}" --array="1-${test_count}" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process and output dir. Eg: extract-frames.sh batchfile.txt out" >&2
        exit 1
    fi
fi
