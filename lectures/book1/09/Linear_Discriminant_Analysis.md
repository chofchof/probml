# 9. Linear Discriminant Analysis

Consider a **classification model** called a **generative classifier**:

$$
p(y=c|x;\theta) = \frac{p(x|y=c;\theta)p(y=c;\theta)}{\sum_{c'}p(x|y=c';\theta)p(y=c';\theta)}
$$



## 9.2 Gaussian discriminant analysis (GDA)

Assume that the **class conditional densities** are multivariate Gaussians, i.e.,

$$
p(x|y=c;\theta) = \mathcal{N}(x|\mu_c,\Sigma_c)
$$

Then the **class posterior** has the form

$$
p(y=c|x;\theta) \propto \pi_c\,\mathcal{N}(x|\mu_c,\Sigma_c) \tag{9.3}
$$

where $\pi_c=p(y=c)$ is the given **class prior**.



### 9.2.4 Model fitting

How to fit a GDA model $p(x,y|\theta)$ using **MLE**:

$$
\begin{align*}
p(\mathcal{D}|\theta) &= \prod_n p(x_n,y_n|\theta) = \prod_n \Bigl[ \operatorname{Cat}(y_n|\pi)\,\prod_c\mathcal{N}(x_n|\mu_c,\Sigma_c)^{\mathbb{I}(y_n=c)} \Bigr] \\
\log p(\mathcal{D}|\theta) &= \sum_c N_c\log\pi_c + \sum_c \sum_{n:y_n=c} \log\mathcal{N}(x_n|\mu_c,\Sigma_c)
\end{align*}
$$

Optimize $\pi_c$ (§4.2.4) and $(\mu_c,\Sigma_c)$ (§4.2.6) separately:

$$
\begin{align*}
\hat\pi_c &= \frac{N_c}{N}, \qquad
\hat\mu_c = \frac{1}{N_c}\sum_{n:y_n=c} x_n, \text{\quad (sample mean)} \\
\hat\Sigma_c &= \frac{1}{N_c}\sum_{n:y_n=c} (x_n-\hat\mu_c)(x_n-\hat\mu_c)^T \text{\quad (empirical covariance)}
\end{align*}
$$

Given a design matrix $X$ of shape $(N,D)=$ `(n_samples, n_features)`, notice that

```python
numpy.cov(X, rowvar=False) # (unbiased) sample covariance
numpy.cov(X, rowvar=False, bias=True) # (biased) empirical covariance
sklearn.covariance.empirical_covariance(X) # the same as above (biased)
```



### 9.2.1 Quadratic decision boundaries

The log posterior over class labels (eq. 9.3) is called the **discriminant function**,

$$
\log p(y=c|x;\theta) \propto \log\pi_c -\frac{1}{2}\log|\Sigma_c|-\frac{1}{2}(x-\mu_c)^T\Sigma_c^{-1}(x-\mu_c) + \text{const} \tag{9.4}
$$

The decision boundary between any two classes will be a <u>quadratic function</u> of $x$. Hence this is known as **quadratic discriminant analysis (QDA)**.

```python
from sklearn import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis(reg_param=0.0)
clf.fit(X, y) # use (unbiased) sample covariance, not biased one
clf._decision_function(X) # discriminant function w/o const (eq. 9.4)
clf.decision_function(X) # same as _decision_function(X)
    # for the binary classification: log p(y=1|x) - log p(y=0|x)
clf.predict(X) # argmax of _decision_function(X)
clf.predict_proba(X) # softmax of _decision_function(X), p(y=c|x)
clf.predict_log_proba(X) # log of predict_proba(X), log p(y=c|x)
```

Let $X_c$ be the (centered) design matrix for the class label $c$ , i.e., the rows of $X_c$ are $(x_n-\hat\mu_c)^T$ for $\{n: y_n=c\}$. Note that $X_c$ is of shape $(N_c,D)$.

Let $X_c=USV^T$ be the (reduced) singular value decomposition. Then the <u>sample covariance</u> for the class label $c$ (`clf.covariance_[c]`) $\hat\Sigma_c$ satisfies

$$
\hat\Sigma_c = \frac{1}{N_c-1} \sum_{n:y_n=c} (x_n-\hat\mu_c)(x_n-\hat\mu_c)^T = \frac{X_c^T X_c}{N_c-1} = \frac{(VSU^T)(USV^T)}{N_c-1} = V\frac{S^2}{N_c-1}V^T
$$

- `clf.scalings_`: eigenvalues of $\hat\Sigma_c$, that is $\frac{S^2}{N_c-1}$ of shape $(D,)$
- `clf.rotations_`: eigenvectors of $\hat\Sigma_c$, that is $V$of shape $(D,D)$

Note that `reg_param` $0\leq\lambda\leq1$ controls `clf.scalings_` using $(1-\lambda)\frac{S^2}{N_c-1}+\lambda$, i.e., $\hat\Sigma_c$ for $\lambda=0$ and the identity matrix $I_D$ for $\lambda=1$.



### Problem: The MLE for $\hat\Sigma_c$ can easily <u>overfit</u> if $N_c$ is small compared to $D$.



### 9.2.2 Linear decision boundaries

If $\Sigma_c=\Sigma$ <u>for all classes</u> $c$, the log posterior over class labels (eq. 9.4) is simplified to a <u>linear function</u> of $x$.

$$
\begin{align*}
\log p(y=c|x;\theta) &\propto \log\pi_c - \frac{1}{2}(x-\mu_c)^T\Sigma^{-1}(x-\mu_c) + \text{const} \tag{9.5} \\
&= \Bigl[\log\pi_c - \frac{1}{2}\mu_c^T\Sigma^{-1}\mu_c \Bigr] + \Bigl[ x^T\Sigma^{-1}\mu_c \Bigr] + \Bigl[ \text{const} - \frac{1}{2}x^T\Sigma^{-1}x \Bigr] \\
&\equiv \gamma_c + x^T\beta_c + \text{const indep. of $c$} \tag{9.7}
\end{align*}
$$

Hence, this method is called **linear discriminant analysis (LDA)**. Note that

$$
\beta_c\equiv\Sigma^{-1}\mu_c \quad\text{and}\quad \gamma_c\equiv\log\pi_c-\frac{1}{2}\mu_c^T\beta_c
$$



#### 9.2.4.1 Tied (or shared) covariance

Use the weighted sum $\sum_c\pi_c\hat\Sigma_c$ of <u>empirical covariances</u> as a shared covariance (`clf.covariance_`) $\hat\Sigma$. In particular, when $\pi_c=\frac{N_c}{N}$ (MLE), we have the <u>within-class scatter</u> matrix $\frac{1}{N}S_W$:

$$
\hat\Sigma \equiv \sum_c \pi_c\hat\Sigma_c = \sum_c \frac{N_C}{N} \hat\Sigma_c = \frac{1}{N} \sum_c \sum_{n:y_n=c} (x_n-\hat\mu_c)(x_n-\hat\mu_c)^T \equiv \frac{1}{N} S_W \tag{9.21}
$$

```python
from sklearn import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
clf.fit(X, y)
# Notice that 'lsqr' solver does not support clf.transform(X)
clf.decision_function(X) # discriminant function w/o const (eq. 9.7)
clf.predict(X) # argmax of decision_function(X)
clf.predict_proba(X) # softmax of decision_function(X)
    # For the binary classification: (1-p, p), where p=sigmoid(decision_function(X))
    # In this case, decision_function(X) is of shape (1,) (see below)
clf.predict_log_proba(X) # log of predict_proba(X)
```

Notice that the `lsqr` solver based on `scipy.linalg.lstsq` always gives a solution regardless of $\hat\Sigma$.

- `clf.coef_`: of shape $(C,D)$. Each row is the <u>least squares solution</u> of $\hat\mu_c=\hat\Sigma\cdot x$, that is, $\beta_c$ (eq. 9.7).
- `clf.intercept_`: of shape $(C,)$ with values $\gamma_c$ (eq. 9.7).

In binary classification, `clf.coef_` is of shape `(1,D)`, that is, `clf.coef_[1, :] - clf.coef_[0, :]`, and `clf.intercept_` is of shape `(1,)`, that is, `clf.intercept_[1] - c.intercept_[0]`.



#### 9.2.4.2 Diagonal covariance

If we force $\hat\Sigma_c$ to be diagonal, we reduce the number of parameters from $O(CD^2)$ to $O(CD)$, which <u>avoids the overfitting problem</u>. However, this <u>loses the ability to capture correlation</u> between the features. (This is known as the **naive Bayes assumption**, see §9.3 and `sklearn.naive_bayes.GaussianNB`.)

We can further restrict the model capacity by using a <u>shared (tied) diagonal covariance matrix</u>, called **diagonal LDA**. (See [HTF09, p.652])

> [HTF09] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2E, 2009.



#### 9.2.4.3 MAP estimation

In high dimensions where $D\gg N$, the <u>MLE estimate for the covariance can easily become singular</u>, and forcing the covariance matrix to be diagonal is a rather strong assumption.

An alternative approach is to perform MAP estimation of a (shared) full covariance Gaussian, rather than using the MLE. The MAP estimate is (see §4.5.2)

$$
\hat\Sigma_\text{map} = \lambda\operatorname{diag}(\hat\Sigma_\text{mle}) + (1-\lambda)\hat\Sigma_\text{mle} = \begin{cases} \hat\Sigma_\text{mle} & \text{for diagonals} \\ (1-\lambda) \hat\Sigma_\text{mle} & \text{for off-diagonals} \end{cases} \tag{9.22}
$$

where $\lambda$ controls the amount of regularization. This technique is known as **regularized discriminant analysis** or **RDA**. (See [HTF09, p.656])

```python
# Note that shrinkage works only with ‘lsqr’ and ‘eigen’ solvers.
LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.1)
LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.9)
```

- `shrinkage=None`: (default) $\hat\Sigma\equiv\hat\Sigma_\text{mle}=\frac{1}{N}S_W$ (eq. 9.21)
- `shrinkage=lambda`: $\hat\Sigma\equiv\hat\Sigma_\text{map}=\lambda\operatorname{diag}(\hat\Sigma_\text{mle}) + (1-\lambda)\hat\Sigma_\text{mle}$ (eq. 9.22)
- `shrinkage='auto'`: Use the **Ledoit-Wolf lemma** (see §4.5.2.1)



### 9.2.5 Nearest centroid classifier

Assume a uniform prior over classes, i.e., $\pi_c=\pi$ for all classes $c$. Then the most probable class label (eq. 9.5) is

$$
\hat y(x) = \operatorname{argmax}_c \log p(y=c|x;\theta) = \operatorname{argmin}_c (x-\mu_c)^T\Sigma^{-1}(x-\mu_c) = \operatorname{argmin}_c d^2(x,\mu_c)
$$

where $d$ is the Mahalanobis distance. This is called the **nearest centroid classifier**, or **nearest class mean classifier (NCM)**.

We can replace $d$ with any other distance metric. One simple approach to learn $d$ is to use (following the notation of [Men+12])

$$
d^2(x,\mu_c) = \|x-\mu_c\|_W^2 = \|W(x-\mu_c)\|^2 = (x-\mu_c)^T(W^TW)(x-\mu_c)
$$

The corresponding class posterior becomes

$$
p(y=c|x;\mu,W) = \frac{\exp(-\frac{1}{2}\|W(x-\mu_c)\|^2)}{\sum_{c'}\exp(-\frac{1}{2}\|W(x-\mu_{c'})\|^2)}
$$

We can optimize $W$ using <u>gradient descent applied to the discriminative loss</u> $\mathcal{L}$ [Men+12] , where

$$
\begin{align*}
\mathcal{L} &= -\frac{1}{N} \sum_n \ln p(y=y_n|x_n) \\
\nabla_W\mathcal{L} &= \frac{1}{N} \sum_n\sum_c \bigl[ \mathbb{I}(y_n=c)-p(y=c|x_n) \bigr] W(x_n-\mu_c)(x_n-\mu_c)^T
\end{align*}
$$

This is called **nearest class mean metric learning**. The advantage of this technique is that it can be used for **one-shot learning** of new classes, since we just need to see a single labeled prototype $\mu_c$ per class (assuming we have learned a good $W$ already).



### Problem: GDA is problematic in <u>high dimensions</u>.



### 9.2.6 Fisher's linear discriminant analysis *

1. <u>Reduce the dimensionality of the features</u> $x\in\mathbb{R}^D$ and then fit an MVN to the resulting low-dimensional features $z\in\mathbb{R}^K$. The simplest approach is to use a linear projection (e.g., PCA) $z=W^Tx$, where $W$ is a $D\times K$ matrix.
2. Use <u>gradient based methods</u> to optimize the log likelihood, derived from the class posterior in the low dimensional space as $\mathcal{L}$ in §9.2.5.
3. (**Fisher's linear discriminant analysis**, or **FLDA**) Find $W$ such that the low-dimensional data can be classified as well as possible using a Gaussian class-conditional density model.

FLDA is an interesting <u>hybrid of discriminative and generative techniques</u>. The drawback is that it is restricted to using $K\leq C-1$ dimensions, regardless of $D$.



#### 9.2.6.2 Extension to higher dimensions and multiple classes [DHS01]

Let $\mu_c=\frac{1}{N_C}\sum_{n:y_n=c} x_n$ be the class mean and $\mu=\frac{1}{N}\sum_c N_c\mu_c=\frac{1}{N}\sum_n x_n$ be the overall mean. Define the **between-class scatter**

$$
S_B\equiv S_T-S_W=\sum_c N_c(\mu_c-\mu)(\mu_c-\mu)^T
$$

where

$$
\begin{align*}
S_T &= \sum_n (x_n-\mu)(x_n-\mu)^T \text{\quad (\textbf{total scatter})} \\
S_W &= \sum_c \sum_{n:y_n=c} (x_n-\mu_c)(x_n-\mu_c)^T \text{\quad (\textbf{within-class scatter})}
\end{align*}
$$

1. $\frac{1}{N}S_T$ is the (total) empirical covariance
2. $\frac{1}{N}S_W=\sum_c\frac{N_c}{N}\hat\Sigma_c$ is the weighted sum of empirical covariances over classes (eq. 9.21)
3. $\frac{1}{N}S_B=\sum_c\frac{N_c}{N} (\mu_c-\mu)(\mu_c-\mu)^T$

The projection from $x\in\mathbb{R}^D$ to $z\in\mathbb{R}^K$ is accomplished by $K\leq C-1$ discriminant functions $z_k=w_k^T x$. If the weight vectors $w_k$ are viewed as the columns of $D\times K$ matrix $W$, then the projection can be written as $z=W^Tx$.

Define $m_c=\frac{1}{N_c}\sum_{n:y_n=c} z_n$ and $m=\frac{1}{N}\sum_c N_c m_c=\frac{1}{N}\sum_n z_n$. Then

$$
\begin{align*}
\tilde S_B &= \sum_c N_c (m_c-m)(m_c-m)^T = W^TS_BW \\
\tilde S_W &= \sum_c\sum_{n:y_n=c} (z_n-m_c)(z_n-m_c)^T = W^TS_WW
\end{align*}
$$

<u>The goal is to find a transformation matrix $W$</u> that maximize the ratio of the between class scatter to the within-class scatter. A simple scalar measure of scatter is the determinant of the scatter matrix

$$
J(W) = \frac{|\tilde S_B|}{|\tilde S_W|} = \frac{|W^TS_BW|}{|W^TS_WW|}
$$

[https://arxiv.org/abs/1903.11240] According to Rayleigh-Ritz quotient method, the optimization problem:
$$
\max_w J(w) \equiv \max_w \frac{w^TS_Bw}{w^TS_Ww}
$$

can be restated as:

$$
\max_w w^TS_Bw \quad\text{subject to}\quad w^TS_Ww=1
$$

The Lagrangian is $\mathcal{L} = w^TS_Bw - \lambda(w^TS_Ww-1)$, where $\lambda$ is the Lagrange multiplier. Then

$$
\frac{\partial\mathcal{L}}{\partial w} = 2S_Bw - 2\lambda S_Ww = 0 \quad\Longrightarrow\quad S_B w = \lambda S_W w
$$

Thus <u>the columns of an optimal $W$are the generalized eigenvectors $w_k$</u> that correspond to the largest eigenvalues $\lambda_k$ in

$$
S_B w_k = \lambda_k S_W w_k
$$

Note that $\tilde S_B=W^TS_BW=\operatorname{diag}(\lambda_k)$ and $\tilde S_W=W^TS_WW=I_K$ so that

$$
J(W) = \frac{|\tilde S_B|}{|\tilde S_W|} = \frac{|W^TS_BW|}{|W^TS_WW|} = \prod_k\lambda_k
$$

In general the solution for $W$ is not unique.

```python
clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None, n_components=K)
clf.fit(X, y)
X_trans = clf.transform(X) # dimension reduction;
    # (X @ clf.scalings_)[:,:n_components]
clf.decision_function(X) # discriminant function w/o const (eq. 9.7)
clf.predict(X) # argmax of decision_function(X)s
clf.predict_proba(X) # softmax of decision_function(X)
    # For the binary classification: (1-p, p), where p=sigmoid(decision_function(X))
    # In this case, decision_function(X) is of shape (1,) (see below)
clf.predict_log_proba(X) # log of predict_proba(X)
```

Notice that the parameter `shrinkage` is involved in calculating $S_W$ and $S_T$ so that $S_B$ too. See §9.2.4.3 above. Notice also that the parameter `n_components` is applied only in the method `clf.transform(X)` by producing the first `n_components` of $z=W^Tx$.

The `eigen` solver based on `scipy.linalg.eigh` may cause the `LinAlgError`, because we cannot guarantee that $S_W$ is positive-definite. We can avoid this problem by adjusting `shrinkage`.

- `clf.scalings_`: $W$ of shape $(D,K)$. Generalized eigenvectors (as columns) that correspond to the `K` largest eigenvalues in $S_Bw_k=\lambda_k S_W w_k$.
- `clf.coef_`: of shape $(C,D)$. Each row is $\beta_c=WW^T\mu_c$ so that $x^T\beta_c=x^TWW^T\mu_c=z^TI_k^{-1}m_c$ (eq. 9.7).
- `clf.intercept_`: of shape $(C,)$ with values $\gamma_c$ so that $\mu_c^T\beta_c=\mu_c^TWW^T\mu_c=m_c^TI_k^{-1}m_c$ (eq. 9.7). 

In binary classification, `clf.coef_` is of shape $(1,D)$, that is, `clf.coef_[1, :] - clf.coef_[0, :]`, and `clf.intercept_` is of shape $(1,)$, that is, `clf.intercept_[1] - c.intercept_[0]`.



#### `scikit-learn`'s linear discriminant analysis *

> The default `svd` solver for `scikit-learn`'s `LinearDiscriminantAnalysis` may be preferable in situations where <u>the number of features is large</u>. It cannot be used with shrinkage because it does not calculate the covariance matrix.
>
> Two singular value decompositions are computed: the SVD of the centered input matrix and the SVD of the class-wise mean vectors.

Let $X_c$ be the (centered) design matrix for the class label $c$ , of shape $(N_c,D)$, and let $X$ be the full design matrix of shape $(N,D)$, i.e., the row concatenation of $X_c$'s.

Denote by $\tilde X$ the <u>standardized</u> input matrix of shape $(N,D)$, i.e., each $d$-column of $X$ is divided by the standard deviation $\sigma_d$ of the inputs in feature $d$ (see §10.2.8).

##### A. First projection $W_1$: Make the (normalized) within-class scatter the identity matrix

Let $\frac{1}{\sqrt{N-C}}\tilde X=U_WS_WV_W^T$ be the (reduced) singular value decomposition. Here, the rank of $S_W$ is denoted by $K_W\leq\min\{N,D\}$, and the <u>(within-class) covariance</u> (of degree of freedom $C$) has the form

$$
\frac{1}{N-C}\tilde X^T\tilde X = V_WS_W^2V_W^T
$$

Define the first projection for (normalized) within-class scatter by $W_1\equiv (V_W/\sigma_d)S_W^{-1}$, that is of shape $(D,K_W)$ and $W_1^Tx=S_W^{-1}(V_W/\sigma_d)^Tx=S_W^{-1}V_W^T(x/\sigma_d)$.

##### B. Second projection $W_2$: Find the singular vectors of the between-class scatter

Let $M$ be the (centered) mean matrix of shape $(C,D)$ with rows $\sqrt{N_c}(\mu_c-\mu)^T$ over classes, and denote by $\tilde M\equiv M\cdot W_1$ the projection of the centered mean matrix.

Let $\frac{1}{\sqrt{C-1}}\bar M=U_BS_BV_B^T$ be the (reduced) singular value decomposition. Here, the rank of $S_B$ is denoted by $K_B\leq\min\{C-1,K_W\}\leq\min\{C-1,N,D\}$, and the <u>(between-class) covariance</u> (of degree of freedom 1) has the form

$$
\frac{1}{C-1}\tilde M^T\tilde M = V_BS_B^2V_B^T
$$

Define the second projection for between-class scatter by $W_2\equiv V_B$, that is of shape $(K_W,K_B)$.

##### C. Total projection $W=W_1\cdot W_2$

Finally define the total projection of shape $(D,K_B)$ by

$$
W\equiv W_1\cdot W_2 = (V_W/\sigma_d)S_W^{-1}\cdot V_B
$$

so that

$$
W^Tx = P_K^T\cdot V_B^T\cdot S_W^{-1}\cdot V_W^T\cdot(x/\sigma_d)
$$

```python
clf = LinearDiscriminantAnalysis(solver='svd', n_components=K) # default
clf.fit(X, y)
X_trans = clf.transform(X) # dimension reduction;
    # ((X - clf.xbar_) @ clf.scalings_)[:,:n_components]
clf.decision_function(X) # discriminant function w/o const (eq. 9.7)
clf.predict(X) # argmax of decision_function(X)s
clf.predict_proba(X) # softmax of decision_function(X)
    # For the binary classification: (1-p, p), where p=sigmoid(decision_function(X))
    # In this case, decision_function(X) is of shape (1,) (see below)
clf.predict_log_proba(X) # log of predict_proba(X)
```

Notice that the `svd` solver <u>does not</u> provide `shrinkage` because it does not calculate covariance. Notice also that the parameter `n_components` is applied only in the method `clf.transform(X)` by projecting $z=W^T(x-\mu)$ on to the first `n_components`.

- `clf.scalings_`: $W$ of shape $(D,K_B)$.
- `clf.coef_`: of shape `(C,D)`. Each row is $\beta_c=WW^T(\mu_c-\mu)$ so that $x^T\beta_c=x^TWW^T(\mu_c-\mu)=z^T(m_c-m)$ (eq. 9.7).
- `clf.intercept_`: of shape `(C,)` with values $\gamma_c$ so that $\mu_c^T\beta_c=(\mu_c-\mu)^T\beta_c+\mu^T\beta_c=\|m_c-m\|^2+\mu^T\beta_c$ (eq. 9.7). 

In binary classification, `clf.coef_` is of shape `(1,D)`, that is, `clf.coef_[1, :] - clf.coef_[0, :]`, and `clf.intercept_` is of shape `(1,)`, that is, `clf.intercept_[1] - c.intercept_[0]`.



## 9.3 Naive Bayes classifiers

##### Naive Bayes assumption

Assume the features are conditionally independent given the class label.

$$
p(x|y=c;\theta) = \prod_d p(x_d|y=c;\theta_{dc})
$$

where $\theta_{dc}$ are the parameters for the class conditional density for class $c$ and feature $d$.



##### Naive Bayes classifier (NBC)

Another **generative classifier** with the naive Bayes assumption:

$$
p(y=c|x;\theta) = \frac{\pi_c \prod_d p(x_d|y=c;\theta_{dc})}{\sum_{c'}\pi_{c'} \prod_d p(x_d|y=c';\theta_{dc'})}
$$



### 9.3.1 Example models

### 9.3.2 Model fitting using MLE

##### A. Binary features, $x_d\in\{0,1\}$ (see §4.2.3)

Use the Bernoulli distribution

$$
p(x|y=c;\theta) = \prod_d\operatorname{Ber}(x_d|\theta_{dc}) \equiv \prod_d \theta_{dc}^{x_d}(1-\theta_{dc})^{1-x_d}
$$

where $\theta_{dc}$ is the probability that $x_d=1$ in class $c$. The MLE is the empirical fraction of 1s,

$$
\hat\theta_{dc} = \frac{N_{dc}}{N_c}
$$

where $N_{dc}=\sum_n\mathbb{I}(x_{nd}=1,y_n=c)$ and $N_c=\sum_n\mathbb{I}(y_n=c)$.



##### B. Categorical features, $x_d\in\{1,\dotsc,K\}$ (see §4.2.4)

Use the categorical distribution

$$
p(x|y=c;\theta) = \prod_d\operatorname{Cat}(x_d|\theta_{dc}) \equiv \prod_d \prod_k \theta_{dck}^{\mathbb{I}(x_d=k)}
$$

where $\theta_{dck}$ is the probability that $x_d=k$ in class $c$. The MLE is the empirical fraction of times event $k$ occurs,

$$
\hat\theta_{dck} = \frac{N_{dck}}{N_c}
$$

where $N_{dck}=\sum_n\mathbb{I}(x_{nd}=k,y_n=c)$.



##### C. Real-valued features, $x_d\in\mathbb{R}$ (see §4.2.5)

Use the univariate Gaussian distribution

$$
p(x|y=c;\theta) = \prod_d\mathcal{N}(x_d|\mu_{dc},\sigma_{dc}^2) \equiv \prod_d \frac{1}{\sqrt{2\pi\sigma_{dc}^2}}\exp\Bigl[ -\frac{1}{2}\Bigl(\frac{x_d-\mu_{dc}}{\sigma_{dc}}\Bigr)^2 \Bigr]
$$

where $\mu_{dc}$ is the mean of feature $d$ in class $c$, and $\sigma_{dc}^2$ is its variance. (<u>This is equivalent to GDA using diagonal covariance matrices</u>.) The MLE is

$$
\hat\mu_{dc} = \frac{1}{N_c}\sum_{n:y_n=c} x_{nd} \quad\text{and}\quad
\hat\sigma_{dc}^2 = \frac{1}{N_c} \sum_{n:y_n=c} (x_{nd}-\hat\mu_{dc})^2
$$



### 9.3.3 Bayesian naive Bayes

> Use the posterior mean $\bar\theta$ instead of the MLE estimate $\hat\theta_\text{mle}$.

From the Bayes rule the posterior has the form

$$
p(\theta|\mathcal{D}) = \frac{p(\theta)p(\mathcal{D}|\theta)}{\sum_{\theta'}p(\theta')p(\mathcal{D}|\theta')} \tag{4.107}
$$

and the most probable value of the parameter is the MAP estimate

$$
\hat\theta_\text{map} = \arg\max_\theta p(\theta|\mathcal{D}) = \arg\max_\theta [\log p(\theta)+\log p(\mathcal{D}|\theta)] \tag{4.118}
$$

The **posterior mode** (MAP estimate) $\hat\theta_\text{map}$ can be a <u>poor summary</u> of the posterior, since it corresponds to a single point. The **posterior mean** $\bar\theta\equiv\mathbb{E}[\theta|\mathcal{D}]$ is a <u>more robust</u> estimate, since it integrates over the whole space.



##### A. Binary features, $x_d\in\{0,1\}$ (see §4.6.2) 

If we multiply the Bernoulli likelihood $p(\mathcal{D}|\theta)=\prod_n\operatorname{Ber}(x_n|\theta)=\theta^{N_1}(1-\theta)^{N_0}$ with the beta prior $p(\theta)\propto\operatorname{Beta}(\theta|\breve\alpha,\breve\beta)$, we get a beta posterior

$$
p(\theta|\mathcal{D}) \propto \operatorname{Beta}(\theta|\breve\alpha+N_1,\breve\beta+N_0) \tag{4.113}
$$

and the posterior mean is given by

$$
\bar\theta \equiv \mathbb{E}[\theta|\mathcal{D}] = \frac{\breve\alpha+N_1}{\breve\alpha+N_1+\breve\beta+N_0} \tag{4.123}
$$

If $\breve\alpha=\breve\beta=0$, this reduces to the MLE, $\hat\theta_{dc}=\frac{N_{dc}}{N_c}$. By contrast, if $\breve\alpha=\breve\beta=1$, we add 1 to all the empirical counts before normalizing. This is called **add-one smoothing** or **Laplace smoothing**.

$$
\bar\theta_{dc} = \frac{1+N_{dc}}{2+N_c}
$$

```python
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha=1.0, binarize=0.0)
clf.fit(X, y)
clf._joint_log_likelihood(X) # _jll(X); see below
clf.predict(X) # argmax of _jll(X)
clf.predict_log_proba(X) # _jll(X) - logsumexp(_jll(X), axis=1)
clf.predict_proba(X) # exp of predict_log_proba(X)
```

Note that the joint log likelihood of a sample $x$ is

$$
\begin{split}
\log p(y=c|x;\theta) &\propto \log\pi_c+\sum_d\log p(x_d|y=c;\theta_{dc}) \\
&= \log\pi_c+\sum_d\Bigl[x_d\log\bar\theta_{dc}+(1-x_d)\log(1-\bar\theta_{dc})\Bigr]
\end{split}
$$

- `clf.class_count`_: Number of samples $N_c$ encountered for each class during fitting, of shape $(C,)$
- `clf.feature_count_`: Number of samples $N_{dc}$ encountered for each class and feature during fitting, of shape $(C,D)$
- `clf.class_log_prior_`: Log probability $\log\pi_c$ of each class, of shape $(C,)$
- `clf.feature_log_prob_`: Empirical log probability of features given a class, $\log\bar\theta_{dc}=\log\frac{\breve\alpha+N_{dc}}{2\breve\alpha+N_c}$, of shape $(C,D)$



##### B. Categorical features, $x_d\in\{1,\dotsc,K\}$ (see §4.6.3) 

We can combine the multinomial likelihood $p(\mathcal{D}|\theta)=\prod_n\operatorname{Cat}(x_n|\theta)=\prod_k\theta_k^{N_k}$ where $N_k=\sum_n\mathbb{I}(x_n=k)$ and Dirichlet prior $p(\theta)\propto\operatorname{Dir}(\theta|\breve\alpha)$ to compute the posterior

$$
p(\theta|\mathcal{D}) \propto \operatorname{Dir}(\theta|\breve\alpha_1+N_1,\dotsc\breve\alpha_K+N_K) \tag{4.156}
$$

and the posterior mean is given by

$$
\bar\theta_k = \frac{\breve\alpha_k+N_k}{\sum_{k'}(\breve\alpha_{k'}+N_{k'})}
\quad\text{or}\quad
\bar\theta_{dck} = \frac{\breve\alpha_{dck}+N_{dck}}{\sum_{k'}(\breve\alpha_{dck'}+N_{dck'})}
\tag{4.158 \& 9.57}
$$

If $\breve\alpha_k=0$ for all $k$, this reduces to the MLE, $\hat\theta_{dck}=\frac{N_{dck}}{N_c}$. Notice that $N_c=\sum_k N_{dck}$ for all $d$.

```python
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB(alpha=1.0)
clf.fit(X, y)
clf._joint_log_likelihood(X) # _jll(X); see below
clf.predict(X) # argmax of _jll(X)
clf.predict_log_proba(X) # _jll(X) - logsumexp(_jll(X), axis=1)
clf.predict_proba(X) # exp of predict_log_proba(X)
```

Note that the joint log likelihood of a sample $x$ is

$$
\begin{align*}
\log p(y=c|x;\theta) &\propto \log\pi_c+\sum_d\log p(x_d|y=c;\theta_{dc}) \\
&= \log\pi_c+\sum_d \sum_k \mathbb{I}(x_d=k)\log\bar\theta_{dck}
\end{align*}
$$

- `clf.category_count_`: List of `ndarray` of length $D$. Each array has the number of samples $N_{dck}$ encountered for each class and category of the specific feature $d$, of shape $(C,K_d)$

- `clf.class_count`_: Number of samples $N_c$ encountered for each class during fitting, of shape $(C,)$

- `clf.n_categories_`: Number of categories $K_d$ for each feature $d$, of shape $(D,)$

- `clf.class_log_prior_`: Log probability $\log\pi_c$ of each class, of shape $(C,)$

- `clf.feature_log_prob_`: List of `ndarray` of length $D$. Each array contains the empirical log probability of the specific feature $d$ given a class, $\log\bar\theta_{dck}=\log\frac{\breve\alpha+N_{dck}}{K_d\breve\alpha+N_c}$, of shape $(C,K_d)$

  

##### C. Real-valued features, $x_d\in\mathbb{R}$

```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB() # no alpha
clf.fit(X, y)
clf._joint_log_likelihood(X) # _jll(X); see below
clf.predict(X) # argmax of _jll(X)
clf.predict_log_proba(X) # _jll(X) - logsumexp(_jll(X), axis=1)
clf.predict_proba(X) # exp of predict_log_proba(X)
```

Note that the joint log likelihood of a sample $x$ is

$$
\begin{align*}
\log p(y=c|x;\theta) &\propto \log\pi_c+\sum_d\log p(x_d|y=c;\theta_{dc}) \\
&= \log\pi_c - \frac{1}{2}\sum_d \Bigl[ \log(2\pi\hat\sigma_{dc}^2)+\Bigl(\frac{x_d-\hat\mu_{dc}}{\hat\sigma_{dc}}\Bigr)^2 \Bigr]
\end{align*}
$$

- `clf.class_count`_: Number of samples $N_c$ encountered for each class during fitting, of shape $(C,)$

- `clf.class_prior_`: Log probability $\log\pi_c$ of each class, of shape $(C,)$

- `clf.theta_`: Mean $\hat\mu_{dc}$ of each feature per class, of shape $(C,D)$

- `clf.var_`: Variance $\hat\sigma_{dc}^2$ of each feature per class, of shape $(C,D)$



##### D. Multinomial naive Bayes (MNB)

The multinomial naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as TF-IDF may also work.

> C.D. Manning, P. Raghavan and H. Schuetze (2008) *Introduction to Information Retrieval*, Cambridge University Press, pp. 234-265. https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

The probability of a document $x$ with word counts $x_d$ being in class $c$ is computed as

$$
p(y=c|x;\theta)\propto \pi_c p(x|y=c;\theta) = \pi_c \prod_d \theta_{dc}^{x_d}
$$

Here, $\theta_{dc}$ is the conditional probability of the $d$-th word occurring in a document of class $c$. We estimate

$$
\hat\theta_{dc} = \frac{N_{dc}}{\sum_{d'}N_{d'c}} \quad\text{and}\quad
\bar\theta_{dc} = \frac{\breve\alpha+N_{dc}}{\sum_{d'}(\breve\alpha+N_{d'c})} = \frac{\breve\alpha+N_{dc}}{D\breve\alpha+\sum_{d'}N_{d'c}}
$$

where $N_{dc}=\sum_{n|y_n=c}x_{nd}$.

```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0)
clf.fit(X, y)
clf._joint_log_likelihood(X) # _jll(X); see below
clf.predict(X) # argmax of _jll(X)
clf.predict_log_proba(X) # _jll(X) - logsumexp(_jll(X), axis=1)
clf.predict_proba(X) # exp of predict_log_proba(X)
```

Note that the joint log likelihood of a sample $x$ is

$$
\begin{align*}
\log p(y=c|x;\theta) &\propto \log\pi_c+\sum_d\log p(x_d|y=c;\theta_{dc}) \\
&= \log\pi_c+\sum_d x_d\log\bar\theta_{dc}
\end{align*}
$$

- `clf.class_count`_: Number of samples $N_c$ encountered for each class during fitting, of shape $(C,)$

- `clf.feature_count_`: Number of samples $N_{dc}$ encountered for each class and feature during fitting, of shape $(C,D)$
- `clf.class_log_prior_`: Log probability $\log\pi_c$ of each class, of shape $(C,)$
- `clf.feature_log_prob_`: Empirical log probability $\log\bar\theta_{dc}$ of features given a class, of shape $(C,D)$



##### E. Complement naive Bayes (CNB)

The complement naive Bayes classifier was designed to correct the <u>severe assumptions</u> made by the standard multinomial naive Bayes classifier. <u>It is particularly suited for imbalanced data sets.</u>

> Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive bayes text classifiers. In ICML (Vol. 3, pp. 616-623). https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

CNB is completely the same as MNB except for using $N_{dc}^c=\sum_{n|y_n\neq c}x_{nd}=\sum_{c'} N_{dc'}-N_{dc}$ instead of $N_{dc}$ so that

$$
\hat\theta_{dc} = \frac{N_{dc}^c}{\sum_{d'}N_{d'c}^c} \quad\text{and}\quad
\bar\theta_{dc} = \frac{\breve\alpha+N_{dc}^c}{\sum_{d'}(\breve\alpha+N_{d'c}^c)} = \frac{\breve\alpha+N_{dc}^c}{D\breve\alpha+\sum_{d'}N_{d'c}^c}
$$

```python
from sklearn.naive_bayes import ComplementNB
clf = ComplementNB(alpha=1.0)
clf.fit(X, y)
clf._joint_log_likelihood(X) # _jll(X); see below
clf.predict(X) # argmax of _jll(X)
clf.predict_log_proba(X) # _jll(X) - logsumexp(_jll(X), axis=1)
clf.predict_proba(X) # exp of predict_log_proba(X)
```

Note that the joint log likelihood of a sample $x$ is

$$
\begin{align*}
\log p(y=c|x;\theta) &\propto \sum_d\log p(x_d|y=c;\theta_{dc}) \\
&= -\sum_d x_d\log\bar\theta_{dc}
\end{align*}
$$

- `clf.class_count`_: Number of samples $N_c$ for each class $c$, of shape $(C,)$
- `clf.feature_all_`: Number of samples $\sum_c N_{dc}$ for each feature $d$, of shape $(D,)$
- `clf.feature_count_`: Number of samples $N_{dc}$ for each class and feature, of shape $(C,D)$
- `clf.feature_log_prob_`: Empirical log probability of features given a class, $-\log\bar\theta_{dc}=-\log\frac{\breve\alpha+N_{dc}^c}{D\breve\alpha+\sum_{d'}N_{d'c}^c}$, of shape $(C,D)$
