POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -m|--modelfile)
    MODELFILE="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--logsdir)
    LOGDIR="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--runmode)
    RUNMODE="$2"
    shift # past argument
    shift # past value
    ;;
    -c|--config)
    CONFIG="$2"
    shift # past argument
    shift # past value
    ;;
    -n|--randomseed)
    RANDOMSEED="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--slice)
    SLICE="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

rm -rf logs/${LOGDIR}
PYTHONPATH=src python src/scripts/rte/da/train_da.py data/fever/fever.db ${CONFIG} logs/${LOGDIR} --cuda-device $CUDA_DEVICE --mode ${RUNMODE} --randomseed ${RANDOMSEED} --slice ${SLICE}
mkdir -p data/models
MODELFILE_NAME=${MODELFILE}_${RANDOMSEED}_${SLICE}
cp logs/${LOGDIR}/model.tar.gz data/models/${MODELFILE_NAME}.tar.gz
echo "Copied model file to data/models/${MODELFILE_NAME}.tar.gz"
PYTHONPATH=src python src/scripts/rte/da/eval_da.py data/fever/fever.db  data/fever/dev.ns.pages.p1.jsonl --param_path ${CONFIG} --randomseed ${RANDOMSEED} --slice ${SLICE}

