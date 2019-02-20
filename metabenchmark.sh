#!/usr/bin/env bash

FRAMEWORKS=(
#constantpredictor
constantpredictor_enc
decisiontree
randomforest

autosklearn
h2oautoml
hyperoptsklearn
oboe
ranger
tpot
)

BENCHMARKS=(
#test
validation
small
chalearn
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
        parallel=5
    else
        parallel=1
    fi
fi

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
        python runbenchmark.py $framework $i -m $mode -p $parallel
    done
elif [[ -z $framework ]]; then
    for i in ${FRAMEWORKS[*]}; do
        python runbenchmark.py $i $benchmark -m $mode -p $parallel
    done
else
    python runbenchmark.py $framework $benchmark -m $mode -p $parallel
fi

