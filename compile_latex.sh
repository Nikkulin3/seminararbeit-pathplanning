#!/bin/bash

function clean() {
    rm *.ac*
    rm *.alg
#    rm *.aux
    rm *.gl*
    rm *.ist
    rm *.klg
    rm *.ko*
    rm *.lo*
    rm *.out
    rm *.run*
    rm *.slg
    rm *.sy*
    rm *.toc
    rm *-blx*
}

function compile() {
    pdflatex -interaction=nonstopmode 000_Main.tex
#    pdflatex -interaction=nonstopmode -halt-on-error 000_Main.tex
}
biber 000_Main
compile
biber 000_Main
compile
compile
#clean
