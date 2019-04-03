#!/usr/bin/env bash

FRAMEWORKS=(
#constantpredictor
#constantpredictor_enc
#decisiontree
randomforest
tunedrandomforest

autosklearn
h2oautoml
oboe
tpot
autoweka
#autoxgboost
#hyperoptsklearn
#ranger
)

BENCHMARKS=(
#test
#validation
small-2c1h
#medium-4c1h
medium-8c8h
)

MODE=(
local
docker
aws
)

mode='local'

usage() {
    echo "Usage: $0 framework_or_benchmark [-m|--mode=<local|docker|aws>]" 1>&2;
}

POSITIONAL=()

for i in "$@"; do
    case $i in
        -h | --help)
            usage
            exit ;;
        -f=* | --framework=*)
            framework="${i#*=}"
            shift ;;
        -b=* | --benchmark=*)
            benchmark="${i#*=}"
            shift ;;
        -m=* | --mode=*)
            mode="${i#*=}"
            shift ;;
        -p=* | --parallel=*)
            parallel="${i#*=}"
            shift ;;
        -*|--*=) # unsupported args
            usage
            exit 1 ;;
        *)
            POSITIONAL+=("$i")
      shift ;;
    esac
done

if [[ -z $parallel ]]; then
    if [[ $mode == "aws" ]]; then
        parallel=40
    else
        parallel=1
    fi
fi

#extra_params="-u /dev/null -o ./stable -Xmax_parallel_jobs=40"
extra_params="-u /dev/null -o ./stable -Xmax_parallel_jobs=40 -Xaws.use_docker=True"

#echo "framework=$framework, benchmark=$benchmark, mode=$mode, extra_params=$extra_params, positional=$POSITIONAL"

#identify the positional param if any
#if [[ -n $POSITIONAL ]]; then
#    if [[ -z $framework ]]; then
#        for i in ${FRAMEWORKS[*]}; do
#fi
#echo "framework=$framework, benchmark=$benchmark, mode=$mode, extra_params=$extra_params, positional=$POSITIONAL"


#run the benchmarks
if [[ -z $benchmark && -z $framework ]]; then
    usage
    exit 1
elif [[ -z $benchmark ]]; then
    for i in ${BENCHMARKS[*]}; do
        python runbenchmark.py $framework $i -m $mode -p $parallel $extra_params
    done
elif [[ -z $framework ]]; then
    for i in ${FRAMEWORKS[*]}; do
        python runbenchmark.py $i $benchmark -m $mode -p $parallel $extra_params
    done
else
    python runbenchmark.py $framework $benchmark -m $mode -p $parallel $extra_params
fi

