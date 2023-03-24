#!/usr/bin/env bash

FRAMEWORKS=(
#constantpredictor
#constantpredictor_enc
#decisiontree
#randomforest
#tunedrandomforest

autosklearn
h2oautoml
tpot
oboe
autoweka
#hyperoptsklearn
#ranger
)

BENCHMARKS=(
#test
#validation
small
medium
large
)

CONSTRAINTS=(
1h8c
4h8c
)

MODE=(
local
docker
aws
)

mode='local'

usage() {
    echo "Usage: $0 framework_or_benchmark [-c|--constraint] [-m|--mode=<local|docker|aws>]" 1>&2;
}

POSITIONAL=()

for i in "$@"; do
    case $i in
        -h | --help)
            usage
            exit ;;
        -f=* | --framework=*)
            frameworks="${i#*=}"
            shift ;;
        -b=* | --benchmark=*)
            benchmarks="${i#*=}"
            shift ;;
        -c=* | --constraint=*)
            constraints="${i#*=}"
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

if [[ -z $frameworks ]]; then
  frameworks=${FRAMEWORKS[*]}
fi

if [[ -z $benchmarks ]]; then
  benchmarks=${BENCHMARKS[*]}
fi

if [[ -z $constraints ]]; then
  constraints=${CONSTRAINTS[*]}
fi

if [[ -z $parallel ]]; then
    if [[ $mode == "aws" ]]; then
        parallel=60
    else
        parallel=1
    fi
fi

#extra_params="-u /dev/null -o ./stable -Xmax_parallel_jobs=40"
extra_params="-u ~/dev/null -o ./stable -Xmax_parallel_jobs=60 -Xaws.use_docker=True -Xaws.query_frequency_seconds=60"
#extra_params="-u ~/.config/automlbenchmark/stable -o ./stable -Xmax_parallel_jobs=20 -Xaws.use_docker=True -Xaws.query_frequency_seconds=60"

#identify the positional param if any
#if [[ -n $POSITIONAL ]]; then
#    if [[ -z $framework ]]; then
#        for i in ${FRAMEWORKS[*]}; do
#fi

#run the benchmarks
#    usage
#    exit 1
for c in ${constraints[*]}; do
    for b in ${benchmarks[*]}; do
        for f in ${frameworks[*]}; do
#            echo "python runbenchmark.py $f $b $c -m $mode -p $parallel $extra_params"
            python runbenchmark.py $f $b $c -m $mode -p $parallel $extra_params
        done
    done
done
