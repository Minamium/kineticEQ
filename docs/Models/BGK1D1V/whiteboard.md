---
title: Whiteboard
parent: BGK1D1V
grand_parent: Models
nav_order: 1
---

## Implicit scheme

\begin{equation}
  f_{i,j}^{k+1}
  = f_{i,j}^{k}
  - \frac{\Delta t}{\Delta x}\, v_{j}\,
    \bigl(
      \hat{f}_{\,i+\tfrac12,j}^{\,k+1}
      -
      \hat{f}_{\,i-\tfrac12,j}^{\,k+1}
    \bigr)
  + \frac{\Delta t}{\tau_i^{k+1}}\,
    \bigl(
      f_{M,i,j}^{k+1}
      -
      f_{i,j}^{k+1}
    \bigr),
  \qquad
  i = 1,\dots,N_x-2,\; j = 0,\dots,N_v-1.
\end{equation}

\begin{equation}
\hat{f}_{\,i+\tfrac12,j}^{\,k+1}
=
\begin{cases}
  f_{\,i,j}^{\,k+1}, & v_j > 0,\\[6pt]
  f_{\,i+1,j}^{\,k+1}, & v_j < 0,
\end{cases}
\qquad
\hat{f}_{\,i-\tfrac12,j}^{\,k+1}
=
\begin{cases}
  f_{\,i-1,j}^{\,k+1}, & v_j > 0,\\[6pt]
  f_{\,i,j}^{\,k+1}, & v_j < 0.
\end{cases}
\end{equation}

\begin{equation}
-\,\beta_j\,f_{\,i+1,j}^{k+1}
+\Bigl(1+\alpha_j+\beta_j+\tfrac{\Delta t}{\tau_i^{k+1}}\Bigr)\,
   f_{\,i,j}^{k+1}
-\,\alpha_j\,f_{\,i-1,j}^{k+1}
=
f_{\,i,j}^{k}
+\frac{\Delta t}{\tau_i^{k+1}}\,
  f_{M,i,j}^{k+1}
\end{equation}

\begin{equation}
\alpha_j=\frac{\Delta t}{\Delta x}\,\max(v_j,0),
\qquad
\beta_j =\frac{\Delta t}{\Delta x}\,\max(-v_j,0).
\end{equation}

\begin{equation}
  \alpha_j \;=\; \frac{\Delta t}{\Delta x}\,\max\!\bigl(v_j,0\bigr),
  \qquad
  \beta_j  \;=\; \frac{\Delta t}{\Delta x}\,\max\!\bigl(-v_j,0\bigr)
\end{equation}

\begin{equation}
\begin{aligned}
  a_{i,j}^{(z)} &= \beta_j,
  &\quad&(i=0,\dots,N_x-2) &\text{(上対角)},\\[4pt]
  b_{i,j}^{(z)} &= 1+\alpha_j+\beta_j+\dfrac{\Delta t}{\tau_i^{(z)}},
  &\quad&(i=0,\dots,N_x-1) &\text{(主対角)},\\[4pt]
  c_{i,j}^{(z)} &= \alpha_j,
  &\quad&(i=1,\dots,N_x-1) &\text{(下対角)}
\end{aligned}
\end{equation}

\begin{equation}
  \mathcal{A}_j^{(z)}
  \;=\;
  \begin{bmatrix}
    b_{0,j}^{(z)}     & -a_{0,j}^{(z)}    &                     &                     \\
    -c_{1,j}^{(z)}    &  b_{1,j}^{(z)}    & -a_{1,j}^{(z)}      &                     \\
                      &  \ddots           &  \ddots             & \ddots              \\
                      &                   & -c_{N_x-2,j}^{(z)}  & b_{N_x-1,j}^{(z)}
  \end{bmatrix}_{N_x\times N_x}
\end{equation}

 変数ベクトル（未知量）
\begin{equation}
  \mathbf{f}^{(z+1)}_{j}
  \;=\;
  \begin{bmatrix}
    f^{(z+1)}_{0,j}\\
    f^{(z+1)}_{1,j}\\
    \vdots\\
    f^{(z+1)}_{N_x-1,j}
  \end{bmatrix},
  \qquad
  j = 0,\dots,N_v-1 .
\end{equation}

 右辺（ソース）ベクトル
\begin{equation}
  \mathbf{d}^{(z)}_{j}
  \;=\;
  \begin{bmatrix}
    d^{(z)}_{0,j}\\
    d^{(z)}_{1,j}\\
    \vdots\\
    d^{(z)}_{N_x-1,j}
  \end{bmatrix},
  \qquad
  d^{(z)}_{i,j}
  \;=\;
  f^{k}_{i,j}
  +\frac{\Delta t}{\tau^{(z)}_{i}}\,
   f_{M,i,j}^{(z)} .
\end{equation}