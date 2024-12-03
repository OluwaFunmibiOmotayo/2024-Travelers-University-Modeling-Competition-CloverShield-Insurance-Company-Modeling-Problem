---
title: '2024 Travelers University Modeling Competition: CloverShield Insurance Company
  Modeling Problem'
author: "Oluwafunmibi Omotayo Fasanya and Augustine Kena Adjei"
date: "December 5, 2024"
output:
  pdf_document: default
  beamer_presentation: default
subtitle: LightGBM Modeling Approach
---

```r
\documentclass{beamer}
\usepackage{graphicx}
\usetheme{Madrid}
\usecolortheme{dolphin}
\usepackage{ragged2e}
\usepackage{subcaption}
\usepackage{amsmath}  % For mathematical symbols and equations
\usepackage{verbatim}
\usepackage{fancyvrb}
\usepackage{color}
\usepackage{amsmath}
\usepackage{tikz}
\usetheme{default}
\usepackage{hyperref}


\title{2024 Travelers University Modeling Competition: \\ CloverShield Insurance Company Modeling Problem}
\subtitle{LightGBM Modeling Approach}
\author{Oluwafunmibi Omotayo Fasanya and Augustine Kena Adjei}
\date{December 5, 2024}

\begin{document}

\frame{\titlepage}

\section{What methods did you consider?}

\begin{frame}{Methods Considered}
\textbf{Tree-Based Models:}
\begin{itemize}
    \item \textbf{Random Forest:} Robust and captures non-linear relationships and interactions.
    \item \textbf{XGBoost:} Gradient boosting algorithm, effective for structured tabular data.
    \item \textbf{LightGBM:} Histogram-based decision tree algorithm for efficient tree construction and reduced memory consumption.
\end{itemize}

\textbf{Zero-Inflated Models:}
\begin{itemize}
    \item \textbf{Zero-Inflated Poisson (ZIP):} Models excess zeros in count data.
    \item \textbf{Zero-Inflated Negative Binomial (ZINB):} Addresses overdispersion and excess zeros.
    \item \textbf{Hurdle Model (Negative Binomial):} Separates zero and non-zero counts.
\end{itemize}
\end{frame}

# Rest of the LaTeX code follows

\end{document}
