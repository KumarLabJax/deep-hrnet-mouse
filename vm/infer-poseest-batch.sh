#!/bin/bash
#
#SBATCH --job-name=infer-poseest-arr
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --qos=inference
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
                echo "${VIDEO_FILE}"
                echo "DUMP OF CURRENT ENVIRONMENT:"
                env
                echo "BEGIN PROCESSING: ${VIDEO_FILE}"
                H5_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v2.h5"
                module load singularity
                singularity run --nv "${ROOT_DIR}/deep-hres-net-2019-06-28.simg" "${VIDEO_FILE}" "${H5_OUT_FILE}"

                # Retry several times if we have to. Unfortunately this is needed because
                # ffmpeg will sporadically give the following error on winter:
                #       ffmpeg: symbol lookup error: /.singularity.d/libs/libGL.so.1: undefined symbol: _glapi_tls_Current
                #
                # You can test this by simply running:
                #       singularity exec --nv deep-hres-net-2019-06-28.simg ffmpeg
                #
                # which will fail about 1 out of 10 times or so. I (Keith) haven't been able to
                # figure out a solution for this except for retrying several times.
                MAX_RETRIES=10
                for (( i=0; i<"${MAX_RETRIES}"; i++ ))
                do
                    if [[ ! -f "${H5_OUT_FILE}" ]]
                    then
                        echo "WARNING: FAILED TO GENERATE OUTPUT FILE. RETRY ATTEMPT ${i}"
                        singularity run --nv "${ROOT_DIR}/deep-hres-net-2019-06-28.simg" "${VIDEO_FILE}" "${H5_OUT_FILE}"
                    fi
                done

                if [[ ! -f "${H5_OUT_FILE}" ]]
                then
                    echo "ERROR: FAILED TO GENERATE OUTPUT FILE WITH NO MORE RETRIES"
                fi

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
        echo "ERROR: you need to provide a batch file to process. Eg: ./infer-poseest-batch.sh batchfile.txt" >&2
        exit 1
    fi
fi
