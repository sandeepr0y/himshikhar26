### Linear Transformation:

A linear transformation is a function between two vector spaces that preserves the operations of vector addition and scalar multiplication. In simpler terms, it's a way to transform vectors (like points on a plane) that keeps lines straight and the origin fixed.

For a transformation $T$ to be linear, it must satisfy two main properties:

1.  **Additivity:** The transformation of the sum of two vectors is equal to the sum of their individual transformations.
    $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$

2.  **Homogeneity:** The transformation of a scaled vector is equal to the scaled transformation of the original vector.
    $T(c\vec{u}) = cT(\vec{u})$, where '$c$' is any scalar (a real number).

---

### Example:

Let's look at the example provided in the image to see these properties in action.
The transformation is defined as $T: \mathbb{R}^2 \to \mathbb{R}^2$ by $T(a_1, a_2) = (2a_1 + a_2, a_1)$.

We need to check if it satisfies the two properties.

Let $\vec{u} = (a_1, a_2)$ and $\vec{v} = (b_1, b_2)$.

**1. Checking Additivity: $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$**

*   First, find the sum of the vectors:
    $\vec{u} + \vec{v} = (a_1 + b_1, a_2 + b_2)$

*   Now, apply the transformation $T$ to this sum:
    $T(\vec{u} + \vec{v}) = T(a_1 + b_1, a_2 + b_2)$
    Using the rule $T(x_1, x_2) = (2x_1 + x_2, x_1)$:
    $T(a_1 + b_1, a_2 + b_2) = (2(a_1 + b_1) + (a_2 + b_2), (a_1 + b_1))$
    $T(\vec{u} + \vec{v}) = (2a_1 + 2b_1 + a_2 + b_2, a_1 + b_1)$

*   Next, find the transformations of the individual vectors:
    $T(\vec{u}) = T(a_1, a_2) = (2a_1 + a_2, a_1)$
    $T(\vec{v}) = T(b_1, b_2) = (2b_1 + b_2, b_1)$

*   Now, add these two transformed vectors:
    $T(\vec{u}) + T(\vec{v}) = (2a_1 + a_2, a_1) + (2b_1 + b_2, b_1)$
    $T(\vec{u}) + T(\vec{v}) = ( (2a_1 + a_2) + (2b_1 + b_2), a_1 + b_1 )$
    $T(\vec{u}) + T(\vec{v}) = (2a_1 + a_2 + 2b_1 + b_2, a_1 + b_1)$

*   Comparing the results, we see that $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$. The additivity property holds.

**2. Checking Homogeneity: $T(c\vec{u}) = cT(\vec{u})$**

*   First, scale the vector $\vec{u}$ by a scalar '$c$':
    $c\vec{u} = c(a_1, a_2) = (ca_1, ca_2)$

*   Now, apply the transformation $T$ to this scaled vector:
    $T(c\vec{u}) = T(ca_1, ca_2)$
    Using the rule $T(x_1, x_2) = (2x_1 + x_2, x_1)$:
    $T(ca_1, ca_2) = (2(ca_1) + (ca_2), ca_1)$
    $T(c\vec{u}) = (2ca_1 + ca_2, ca_1)$

*   Next, find the transformation of the original vector $\vec{u}$ and then scale it by '$c$':
    $T(\vec{u}) = (2a_1 + a_2, a_1)$
    $cT(\vec{u}) = c(2a_1 + a_2, a_1)$
    $cT(\vec{u}) = (c(2a_1 + a_2), c(a_1))$
    $cT(\vec{u}) = (2ca_1 + ca_2, ca_1)$

*   Comparing the results, we see that $T(c\vec{u}) = cT(\vec{u})$. The homogeneity property also holds.

Since both properties are satisfied, the transformation $T(a_1, a_2) = (2a_1 + a_2, a_1)$ is a linear transformation.

---

### Checking for Linear Transformation: Rotation by an Angle $\theta$

Now let's consider the transformation $T: \mathbb{R}^2 \to \mathbb{R}^2$ that rotates a vector $(a_1, a_2)$ counter-clockwise by an angle $\theta$.
The rule for this transformation is:
*   If $(a_1, a_2) \neq (0,0)$, $T(a_1, a_2) = (a_1 \cos \theta - a_2 \sin \theta, a_1 \sin \theta + a_2 \cos \theta)$.
*   If $(a_1, a_2) = (0,0)$, $T(0,0) = (0,0)$.

Let's check the two properties of linear transformations.

Let $\vec{u} = (a_1, a_2)$ and $\vec{v} = (b_1, b_2)$.

**1. Checking Additivity: $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$**

*   Sum of vectors:
    $\vec{u} + \vec{v} = (a_1 + b_1, a_2 + b_2)$

*   Apply $T$ to the sum. The resulting vector's components will be $(a_1+b_1)$ and $(a_2+b_2)$ in the positions of $a_1$ and $a_2$ in the transformation rule:
    $T(\vec{u} + \vec{v}) = ((a_1+b_1) \cos \theta - (a_2+b_2) \sin \theta, (a_1+b_1) \sin \theta + (a_2+b_2) \cos \theta)$
    Let's expand this:
    $T(\vec{u} + \vec{v}) = (a_1 \cos \theta + b_1 \cos \theta - a_2 \sin \theta - b_2 \sin \theta, a_1 \sin \theta + b_1 \sin \theta + a_2 \cos \theta + b_2 \cos \theta)$
    We can rearrange the terms:
    $T(\vec{u} + \vec{v}) = ( (a_1 \cos \theta - a_2 \sin \theta) + (b_1 \cos \theta - b_2 \sin \theta), (a_1 \sin \theta + a_2 \cos \theta) + (b_1 \sin \theta + b_2 \cos \theta) )$

*   Now, transform each vector individually:
    $T(\vec{u}) = (a_1 \cos \theta - a_2 \sin \theta, a_1 \sin \theta + a_2 \cos \theta)$
    $T(\vec{v}) = (b_1 \cos \theta - b_2 \sin \theta, b_1 \sin \theta + b_2 \cos \theta)$

*   Add the transformed vectors:
    $T(\vec{u}) + T(\vec{v}) = (a_1 \cos \theta - a_2 \sin \theta, a_1 \sin \theta + a_2 \cos \theta) + (b_1 \cos \theta - b_2 \sin \theta, b_1 \sin \theta + b_2 \cos \theta)$
    $T(\vec{u}) + T(\vec{v}) = ( (a_1 \cos \theta - a_2 \sin \theta) + (b_1 \cos \theta - b_2 \sin \theta), (a_1 \sin \theta + a_2 \cos \theta) + (b_1 \sin \theta + b_2 \cos \theta) )$

*   Comparing the expanded $T(\vec{u} + \vec{v})$ with $T(\vec{u}) + T(\vec{v})$, we see they are the same. The additivity property holds.

**2. Checking Homogeneity: $T(c\vec{u}) = cT(\vec{u})$**

*   Scale the vector $\vec{u}$:
    $c\vec{u} = (ca_1, ca_2)$

*   Apply $T$ to the scaled vector:
    $T(c\vec{u}) = T(ca_1, ca_2)$
    Substitute $ca_1$ for $a_1$ and $ca_2$ for $a_2$ in the transformation rule:
    $T(ca_1, ca_2) = ((ca_1) \cos \theta - (ca_2) \sin \theta, (ca_1) \sin \theta + (ca_2) \cos \theta)$
    Factor out '$c$':
    $T(c\vec{u}) = (c(a_1 \cos \theta - a_2 \sin \theta), c(a_1 \sin \theta + a_2 \cos \theta))$

*   Transform the original vector $\vec{u}$ and then scale it by '$c$':
    $T(\vec{u}) = (a_1 \cos \theta - a_2 \sin \theta, a_1 \sin \theta + a_2 \cos \theta)$
    $cT(\vec{u}) = c(a_1 \cos \theta - a_2 \sin \theta, a_1 \sin \theta + a_2 \cos \theta)$
    $cT(\vec{u}) = (c(a_1 \cos \theta - a_2 \sin \theta), c(a_1 \sin \theta + a_2 \cos \theta))$

*   Comparing the results, we see that $T(c\vec{u}) = cT(\vec{u})$. The homogeneity property also holds.

Since both the additivity and homogeneity properties are satisfied, the rotation by an angle $\theta$ is a linear transformation.

Let's confirm the origin point. For any linear transformation, $T(\vec{0}) = \vec{0}$.
In this case, $\vec{0} = (0,0)$.
$T(0,0) = (0 \cos \theta - 0 \sin \theta, 0 \sin \theta + 0 \cos \theta) = (0,0)$.
This is consistent with the definition given ($T(0,0)=(0,0)$), and it's a necessary condition for linearity.

Therefore, a rotation by any angle $\theta$ is a linear transformation.


A reflection is a type of linear transformation that flips a vector across a line or a plane. It's like looking at yourself in a mirror.

**Example: Reflection across the x-axis**

Let's consider a transformation $T: \mathbb{R}^2 \to \mathbb{R}^2$ that reflects a point $(x, y)$ across the x-axis. The coordinates of the reflected point are $(x, -y)$.

To confirm this is a linear transformation, we check the two properties:

1.  **Additivity:** $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$
    Let $\vec{u} = (x_1, y_1)$ and $\vec{v} = (x_2, y_2)$.
    *   $T(\vec{u} + \vec{v}) = T(x_1 + x_2, y_1 + y_2) = (x_1 + x_2, -(y_1 + y_2)) = (x_1 + x_2, -y_1 - y_2)$.
    *   $T(\vec{u}) = (x_1, -y_1)$ and $T(\vec{v}) = (x_2, -y_2)$.
    *   $T(\vec{u}) + T(\vec{v}) = (x_1, -y_1) + (x_2, -y_2) = (x_1 + x_2, -y_1 - y_2)$.
    Since $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$, the additivity property holds.

2.  **Homogeneity:** $T(c\vec{u}) = cT(\vec{u})$
    Let $\vec{u} = (x, y)$ and $c$ be a scalar.
    *   $T(c\vec{u}) = T(cx, cy) = (cx, -(cy)) = (cx, -cy)$.
    *   $cT(\vec{u}) = c(x, -y) = (cx, -cy)$.
    Since $T(c\vec{u}) = cT(\vec{u})$, the homogeneity property holds.

Because both properties are satisfied, reflection across the x-axis is a linear transformation.

---

### Matrix Representation of Linear Transformations

Any linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ can be represented by a matrix. This matrix allows us to perform the transformation by matrix multiplication.

To find the matrix for a linear transformation, we look at how it transforms the standard basis vectors. The standard basis vectors in $\mathbb{R}^n$ are:
$e_1 = (1, 0, \dots, 0)$
$e_2 = (0, 1, \dots, 0)$
...
$e_n = (0, 0, \dots, 1)$

If $T(e_j)$ is the image of the $j$-th standard basis vector under the transformation $T$, then the matrix representation of $T$, let's call it $A$, will have the vectors $T(e_1), T(e_2), \dots, T(e_n)$ as its columns. So, $A = [T(e_1) | T(e_2) | \dots | T(e_n)]$.
This is an $m \times n$ matrix. For any vector $\vec{x}$ in $\mathbb{R}^n$, the transformation can be calculated as $T(\vec{x}) = A\vec{x}$.

---

### Solving the Sum: $T(x,y) \to (x,0)$

This transformation takes a vector $(x,y)$ and maps it to $(x,0)$. This is a projection onto the x-axis.

**1. How the projection transformation of the matrix would look like:**

We need to find the matrix that performs this transformation. We do this by applying the transformation to the standard basis vectors of $\mathbb{R}^2$, which are $e_1 = (1,0)$ and $e_2 = (0,1)$.

*   Apply $T$ to $e_1 = (1,0)$:
    $T(1,0) = (1,0)$. This will be the first column of our matrix.

*   Apply $T$ to $e_2 = (0,1)$:
    $T(0,1) = (0,0)$. This will be the second column of our matrix.

So, the matrix $A$ for this transformation is:
$A = \begin{bmatrix} T(1,0) & T(0,1) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

**2. What would be the matrix of projection on the point (x,0):**

The phrasing "projection on the point (x,0)" is a bit unusual. Typically, we project onto a line or a plane.
The transformation $T(x,y) \to (x,0)$ is indeed a **projection onto the x-axis**. The point $(x,0)$ is a general point on the x-axis.

The matrix we found, $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$, is precisely the matrix for the linear transformation that projects any vector $(x,y)$ in $\mathbb{R}^2$ onto the x-axis.

Let's verify this with an example. Take the vector $(3, 4)$.
Using the transformation rule: $T(3,4) = (3,0)$.
Using the matrix:
$\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} (1 \times 3) + (0 \times 4) \\ (0 \times 3) + (0 \times 4) \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$

Both methods yield the same result, confirming that $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ is the matrix for projecting a vector $(x,y)$ onto the x-axis, resulting in the point $(x,0)$.
