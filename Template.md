



# 算法模板

## 结论

### 异或

* 给定一个序列$S$,元素的异或和小于$2\times max(S)$
* 一个有序序列中,两个元素的异或值最小一定是两个相邻元素
* $a - b\le a\oplus b \le a + b$

## 杂

### 关闭同步流

``` c++
ios::sync_with_stdio(false);
cin.tie(NULL),cout.tie(NULL);
```

### 快读快写

``` c++
template<class T> inline void read(T &k) {
    T x = 0, f = 1; 
    char ch = getchar(); 
    while (ch < '0' || ch > '9') {
        if (ch == '-') {
            f = -1;
        }
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = (x << 1) + (x << 3) + (ch ^ 48);
        ch = getchar();
    }
    k = x * f;
}
template<class T> inline void write(T x) {
    if (x < 0) {
        putchar('-'), x = -x;
    }
    if (x > 9) {
        write(x / 10);
    }
    putchar(x % 10 + '0');
}
```

### 取整

**有负数情况的取整**

```c++
template <typename T, typename U>
T ceil(T x, U y) {
  return (x > 0 ? (x + y - 1) / y : x / y);
}
template <typename T, typename U>
T floor(T x, U y) {
  return (x > 0 ? x / y : (x - y + 1) / y);
}
```

### bitset与位运算

**位运算的转化**

$a \oplus b= a(1-b)+b(1-a)$

**位运算函数**

``` c++
__builtin_popcount(x)  //x二进制里 1 的个数（unsigned int)
__builtin_popcountll(x) //x二进制里 1 的个数 (unsigned long long)
__builtin_parity(x) // 1的个数的奇偶性
...ctz(x) // x二进制末尾0的个数
...clz(x) // x二进制开头0的个数   
```

**bitset**

```c++
	bitset<8> a(5); //0101
	bitset<8> b(2); //0010
	
	cout << a.count() << endl; //统计1的个数
	cout << a.any() << endl; //判断是否有1
	cout << a.none() << endl; //判断是否全为0
	
	
	a.set(1); //set(x)把x位变成1
	a.reset(1); //reset(x)把x位变成0
```

**枚举子集**

```c++
for (int j = i; j = (j - 1) & i;) {
	f[i] = max(f[i], f[j] + f[i ^ j]);
}
```

### 重载优先队列

```c++
struct cmp
{
    bool operator()(const node &a, const node &b){
        return a.val > b.val;
    }
};
priority_queue<node, vector<node>, cmp> pq;
```



### cnt中选k个的dfs

```c++
//cnt中选k个的dfs
function<void(int, int)> dfs = [&](int x, int y) {
    if (y > k) return;
    if (y == k) {
        ans = max(ans, work());
        return;
    }
    for (int i = x + 1; i <= cnt; ++i) {
        mp[s[i]] = true;
        dfs(i, y + 1);
        mp[s[i]] = false;
    }
};

```

### 模拟

#### 判断两图案是否能通过平移或旋转得到

```c++
auto rotate = [&](vector<string> a) {
    vector<string> res(n, string(n, '.'));
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < n; j ++) {
            res[j][n - 1 - i] = a[i][j];
		}
	}
    return res;
};
auto normalize = [&](vector<string> a) {
}
```

## dp

### 数位dp模板

```c++
//pos表示到哪一位了
//lim代表前面的数字是否都是n对应位上的，如果位true, 那么当前位最多num[i]，否则为9
//lead代表前面是否填了数字，用来处理
int dfs(int pos, int lim, int lead) {
	i64 o = f[][][][][];
	if (o != -1 && !lim) return o;
	if (!pos) return ...;
	i64 res = 0;
	int up = lim ? num[pos] : 9;
	for (int i = 0; i <= up; i ++) {
        ....
	}
	if (!lim && !lead) {
		f[][][][][] = res;
	}
	return res;
}

```



## 树上问题

### 树的重心

定义1：对于$n$个节点的无根树，找到一个点，使得把树变成该点为根树时，最大子树的节点数最小。即删除这个点后最大联通快的节点数最小，那么这个点就是树的重心。

定义2：一颗具有$n$个结点的无根树，若以某个结点为整个树的根，它的每个儿子的子树大小都小于等于$n/2$，则称这个点为该树的重心。

``` c++
vector<int> g[N];
int d[N];
int f[N];
int siz[N];
int pre[N];

void solve(int u, int fa){
    siz[u] = 1;
    for(auto v : g[u]){
        if(v == fa) continue;
        pre[v] = u;
        solve(v , u);
        siz[u] += siz[v];
    }
}

signed main(){
    int n; cin >> n;
    for (int i = 1;i <= n - 1; i++) {
        int u , v , w;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    pre[1] = -1;
    solve(1 , -1);

    int idx = 0;
    int mn = 1e10;
    for (int i = 1;i <= n; i++) {
        int tmp = 0;
        for(auto v : g[i]){
            if(v == pre[i]){
                tmp = max(tmp , n - siz[i]);
            }else{
                tmp = max(tmp , siz[v]);
            }
        }
        if(tmp < mn){
            mn = tmp;
            idx = i;
        }
    }
    cout << idx;

    return 0;
}
```

### 最近公共祖先（LCA）

``` c++
const int N = 2e5 + 5;
int n, dep[N], fa[N][21];
vector<int>g[N];
void dfs_lca(int u, int father) {
    dep[u] = dep[father] + 1;
    fa[u][0] = father;
    for (int i = 1; (1 << i) <= dep[u]; i++) {
        fa[u][i] = fa[fa[u][i - 1]][i - 1];
    }
    for (int child : g[u]) {
        if (child != father) {
            dfs_lca(child, u);
        }
    }
}
int LCA(int a, int b) {
    if (dep[a] > dep[b]) {
        swap(a, b);
    }
    for (int i = 20; i >= 0; i--) {
        if (dep[a] <= dep[b] - (1 << i)) {
            b = fa[b][i];
        }
    }
    if (a == b) {
        return a;
    }
    for (int i = 20; i >= 0; i--) {
        if (fa[a][i] == fa[b][i])continue;
        else {
            a = fa[a][i];
            b = fa[b][i];
        }
    }
    return fa[a][0];
}
//树上两点间的距离
int dis(int u, int v) {
    return dep[u] + dep[v] - 2 * dep[LCA(u, v)];
}
//判断某点是否在两点的最短路径上
bool on(int i, int u, int v) {
  return dis(i, u) + dis(i, v) == dis(u, v);  
}
```

### 树上启发式合并

**主要思路：将轻儿子上的信息忘重儿子上合并**

 题意:给树的节点染色，子树中出现最多次的颜色（可能有多个）称为占领该子树，对每个节点,

 求占领该节点所对应子树的颜色的编号之和。

```c++
#include<bits/stdc++.h>
#define x first
#define y second
#define endl '\n'
using namespace std;
using i64 = int64_t;
const int N = 2e5 + 10;
int n;
vector<int> g[N];
int c[N];
int l[N], r[N], id[N], cnt; //dfs序列
int siz[N]; //子树的大小
int hs[N]; //重儿子
int cnt[N];//每个颜色的出现次数
int ans[N];
int maxcnt;
int sumcnt;
void dfs_init(int u, int fa) {
    l[u] = ++tot;
    id[tot] = u;	
    sz[u] = 1;
    hs[u] = -1;
    for (auto v : g[u]) {
        if (v != fa) {
            dfs_init(v , u);
            sz[u] += sz[v];
            if (hs[u] == -1 || sz[v] > sz[hs[u]]) hs[u] = v;
        }
    }
    r[u] = tot;
}

void dfs_solve(int u, int fa, bool keep) {
    //遍历轻儿子
    for (auto v : g[u]) {
        if (v != fa && v != hs[u]) {
            dfs_solve(v , u, false);
        }
    }
    //遍历重儿子
    if  (hs[u] != -1) {
        dfs_solve(hs[u], u, true);
    }
    //添加操作
    auto add = [&](int x) {
        x = c[x];x
        cnt[x] ++;
        if (cnt[x] > maxcnt) maxcnt = cnt[x] , sumcnt = 0;
        if (cnt[x] == maxcnt) sumcnt += x;        
    };

    //删除操作
    auto del = [&](int x) {
        x = c[x];
        cnt[x] --;
    };

    for (auto v : g[u]) {
        if  (v != fa && v != hs[u]) {
            for (int i = l[v]; i <= r[v]; i ++) {
                add(id[i]);
            }
        }
    }
    add(u);
    ans[u] = sumcnt;
    //清空操作
    if (!keep) {
        maxcnt = 0;
        sumcnt = 0;
        for (int i = l[u]; i <= r[u]; i ++) {
            del(id[i]);
        }
    }

}
signed main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    cin >> n;

    for (int i = 1; i <= n; i ++) cin >> c[i];
    for (int i = 1; i <= n - 1; i ++) {
        int u , v; cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    dfs_init(1 , -1);
    dfs_solve(1 ,-1 ,false);
    
    for (int i = 1; i <= n; i ++) {
        cout << ans[i] << ' ';
    }

    return 0;
}
```



### 树上差分和前缀和

**如果题面多次询问树上的一些路径权值和，就要考虑树上前缀和。**



#### 点前缀和

设$s[i]$表示从根节点到节点$i$的点权和。

自顶向下计算出前缀和$s[i]$，然后用前缀和拼凑$(x,y)$的路径和。

​			$$\large s[x] + s[y] - s[lca(x, y)] - s[fa[lca(x, y)]]$$

#### 边前缀和

设$s[i]$表示从根节点到节点$i$的边权和。

自顶向下计算出前缀和$s[i]$，然后用前缀和拼凑$(x,y)$的路径和。

​		$$\large s[x] + s[y] - 2 * s[lca(x, y)]$$



**如果题面多次对树上的一些路径做加法操作，然后询问某个点或某条边经过操作后的值，就要考虑树上差分。**



#### 点差分

做以下操作

$$\large d[x] + 1,d[y] + 1, d[lca(x, y)] - 1, fa[d[lca(x, y)]] - 1$$

#### 边差分

做以下操作

$$\large d[x] + 1,d[y] + 1,d[lac(x, y)] - 2$$



## 数论

**常用的转换**

$\large k\bmod i = k-i\times \lfloor\frac{k}{i}\rfloor$

$\large d(ij)=\sum_{x|i}\sum_{y|j}[gcd(i,j)=1]$

**常用的结论** 

$\large gcd(2^{i}-1,2^{j}-1)=2^{gcd(i,j)-1}$					

$\large \sum_{i=1}^{n}\sum_{j=1}^{n}[gcd(i,j)=1]=2\sum_{i=1}^{n}\varphi(i)-1$							

**常用求和式转非求和式**

$\large\sum_{i=1}^{n}i=\frac{1}{2}n(n+1)$

$\large\sum_{i=1}^{n}i^2=\frac{1}{6}n(n+1)(2n+1)$

$\large\sum_{i=1}^{n}i^3=\frac{1}{4}n^2(n+1)^2$



| $n<=$  | $10^{1}$  | $10^{2}$  | $10^{3}$  | $10^{4}$  | $10^{5}$  | $10^{6}$  | $10^{7}$  | $10^{8}$  | $10^{9}$  |
| :----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| $w(n)$ |    $2$    |    $3$    |    $4$    |    $5$    |    $6$    |    $7$    |    $8$    |    $8$    |    $9$    |
| $d(n)$ |    $4$    |   $12$    |   $32$    |   $64$    |   $128$   |   $240$   |   $448$   |   $768$   |  $1344$   |
| $n<=$  | $10^{10}$ | $10^{11}$ | $10^{12}$ | $10^{13}$ | $10^{14}$ | $10^{15}$ | $10^{16}$ | $10^{17}$ | $10^{18}$ |
| $w(n)$ |   $10$    |   $10$    |   $11$    |   $12$    |   $12$    |   $13$    |   $13$    |   $14$    |   $15$    |
| $d(n)$ |  $2304$   |  $4032$   |  $6720$   |  $10752$  |  $17280$  |  $26880$  |  $41472$  |  $64512$  | $103680$  |



### 欧拉筛（线性筛法）

$\large O(n)$

#### 筛质数

```c++
std::vector<int> minp, primes;
void sieve(int n) {
    minp.assign(n + 1, 0);
    primes.clear();
    
    for (int i = 2; i <= n; i++) {
        if (minp[i] == 0) {
            minp[i] = i;
            primes.push_back(i);
        }
        
        for (auto p : primes) {
            if (i * p > n) {
                break;
            }
            minp[i * p] = p;
            if (p == minp[i]) {
                break;
            }
        }
    }
}

```

#### 筛约数个数

```c++
int primes[N], cnt; // primes[]存储所有素数
bool vis[N];         // vis[x]存储x是否被筛掉
int d[N];           // d[x]表示x的约数个数
int num[N];         // num[x]表示x的最小质因数的个数
int n;
void seive(int n) {
    d[1] = 1; // 1的约数只有1个,这个比较特殊

    for (int i = 2; i <= n; i++) {
        if (!vis[i]) {
            primes[++cnt] = i;
            // i是质数
            d[i] = 2;   //约数个数是2个，一个是1，另一个是i
            num[i] = 1; //最小质因子个数是1，最小质因子就是自己i
        }

        for (int j = 1; i * primes[j] <= n; j++) {
            st[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                d[i * primes[j]] = d[i] / (num[i] + 1) * (num[i] + 2);
                num[i * primes[j]] = num[i] + 1;
                break;
            } else {
                // d[i * primes[j]] = d[i] * d[primes[j]]; 等价于下面的代码　
                d[i * primes[j]] = d[i] * 2;
                num[i * primes[j]] = 1;
            }
        }
    }
}
```

#### 筛约数的和

```c++
int primes[N], cnt; // primes[]存储所有素数
bool vis[N];         // vis[x]存储x是否被筛掉
int sd[N];           // 约数和
int num[N];         // 最小质因子p1组成的等比序列 p1^0+p1^1+...+p1^r1
int n;
void seive(int n) {
    sd[1] = 1; // 1的约数只有自己，约数和是1

    for (int i = 2; i <= n; i++) {
        if (!vis[i]) {
            primes[++cnt] = i;
            // i是质数
           	sd[i] = num[i] = i + 1;
        }

        for (int j = 1; i * primes[j] <= n; j++) {
            st[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                sd[i * primes[j]] = sd[i] / num[i] * (num[i] * primes[j] + 1);
                num[i * primes[j]] = num[i] * primes[j] + 1;
                break;
            } else {
                sd[i * primes[j]] = sd[i] * sd[primes[j]]; //积性函数
                num[i * primes[j]] = primes[j] + 1;
            }
        }
    }
}
```





### 杜教筛

**解决问题**

​				$\large设f(n)是一个数论函数，计算S(n)=\sum_{i=1}^{n}f(i)$

**杜教筛公式**

​				$\large g(1)S(n)=\sum_{i=1}^{n}h(i)-\sum_{i=2}^{n}g(i)S(\lfloor \frac{n}{i} \rfloor)$



#### 欧拉函数前缀和

​				$\large 令h=id,g=I带入公式得$

​				$\large S(n)=\frac{n(n + 1)}{2}-\sum_{i=2}^{n}S(\frac{n}{i})$

#### 莫比乌斯函数前缀和

​				$\large 令h=\epsilon,g=I带入公式得$

​				$\large S(n)=1-\sum_{i=2}^{n}S(\frac{n}{i})$

```c++
const int N = 5e6 + 7; //超过n^(2/3)就行
int prime[N + 10];  //记录素数
bool vis[N + 10];    //记录是否被筛
i64 mu[N + 10];     //莫比乌斯值
i64 phi[N + 10];    //欧拉函数值
unordered_map<int, int> summu; //莫比乌斯前缀和
unordered_map<int, i64> sumphi; //欧拉函数前缀和

void init() {
    int cnt = 0;
    vis[0] = vis[1] = true;
    phi[1] = mu[1] = 1;
    for (int i = 2; i < N; i ++) {
        if (!vis[i]) {
            prime[++cnt] = i;
            phi[i] = i - 1;
            mu[i] = -1;
        }
        
        for (int j = 1; j <= cnt && i * prime[j] < N; j ++) {
            vis[i * prime[j]] = true;
            if (i % prime[j]) {
                mu[i * prime[j]] = -mu[i];
                phi[i * prime[j]] = phi[i] * (prime[j] - 1);
            } else {
                mu[i * prime[j]] = 0;
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
        }
    }
    for (int i = 1; i < N; i ++) {
        phi[i] += phi[i - 1];
        mu[i] += mu[i - 1];
    }
}
i64 getsummu(i64 x) {
    if (x < N) return mu[x];
    if (summu[x]) return summu[x];
    i64 ans = 1;
    for (i64 l = 2, r; l <= x; l = r + 1) { //利用整除分块
        i64 o = x / l;
        r = x / o;
        ans -= (i64)(r - l + 1) * getsummu(o); 
    }
    return summu[x] = ans;
}
i64 getphisum(i64 x) {
    if (x < N) return phi[x];
    if (sumphi[x]) return sumphi[x];
    i64 ans = (i64)x * (x + 1) / 2;
    for (i64 l = 2, r; l <= x; l = r + 1) {
        i64 o = x / l;
        r = x / o;
        ans -= (i64)(r - l + 1) * getphisum(o);
    }   
    return sumphi[x] = ans;
}

```



### Pollard_rho

$$\large O(n^{1/4} )$$

**求解n的所有质因子**

```c++
using i64 = long long;
i64 mul(i64 a, i64 b, i64 m) {
    return static_cast<__int128>(a) * b % m;
}
i64 power(i64 a, i64 b, i64 m) {
    i64 res = 1 % m;
    for (; b; b >>= 1, a = mul(a, a, m))
        if (b & 1)
            res = mul(res, a, m);
    return res;
}
bool isprime(i64 n) {
    if (n < 2)
        return false;
    static constexpr int A[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    int s = __builtin_ctzll(n - 1);
    i64 d = (n - 1) >> s;
    for (auto a : A) {
        if (a == n)
            return true;
        i64 x = power(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool ok = false;
        for (int i = 0; i < s - 1; ++i) {
            x = mul(x, x, n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok)
            return false;
    }
    return true;
}
std::vector<i64> factorize(i64 n) {
    std::vector<i64> p;
    std::function<void(i64)> f = [&](i64 n) {
        if (n <= 10000) {
            for (int i = 2; i * i <= n; ++i)
                for (; n % i == 0; n /= i)
                    p.push_back(i);
            if (n > 1)
                p.push_back(n);
            return;
        }
        if (isprime(n)) {
            p.push_back(n);
            return;
        }
        auto g = [&](i64 x) {
            return (mul(x, x, n) + 1) % n;
        };
        i64 x0 = 2;
        while (true) {
            i64 x = x0;
            i64 y = x0;
            i64 d = 1;
            i64 power = 1, lam = 0;
            i64 v = 1;
            while (d == 1) {
                y = g(y);
                ++lam;
                v = mul(v, std::abs(x - y), n);
                if (lam % 127 == 0) {
                    d = std::gcd(v, n);
                    v = 1;
                }
                if (power == lam) {
                    x = y;
                    power *= 2;
                    lam = 0;
                    d = std::gcd(v, n);
                    v = 1;
                }
            }
            if (d != n) {
                f(d);
                f(n / d);
                return;
            }
            ++x0;
        }
    };
    f(n);
    std::sort(p.begin(), p.end());
    return p;
}
```



### GCD&EXGCD

#### GCD

$$O(\log(n))$$

``` c++
int gcd(int a, int b) {
    return b ? gcd(b , a % b) : a;
}

<algorithm>里的_gcd
```

#### EXGCD

$$O(\log(n))$$

``` c++
int exgcd(int a, int b, int &x, int &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }

    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
```

##### EXGCD的通解

$\Large x=x_{0}+\frac{kb}{gcd(a,b)},y=y_{0}-\frac{ka}{gcd(a,b)}$

$\large 例题:$

$\large 给定整数N,A,B,C,X.找出满足下面条件三元组(i,j,k)的数量$

* $\large 1\le i,j,k\le N$
* $\large Ai+Bj+Ck=X$

$\large 思路:$

$\large 枚举i=[1,N]的每一个值,满足Bj+Ck=X-Ai,算出每个通解中t的范围$

```c++
#include<bits/stdc++.h>
#define x first
#define y second
#define endl '\n'
using namespace std;
using i64 = int64_t;
using i128 = __int128_t;
template<class T> inline void read(T &k) {
    T x = 0, f = 1; 
    char ch = getchar(); 
    while (ch < '0' || ch > '9') {
        if (ch == '-') {
            f = -1;
        }
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = (x << 1) + (x << 3) + (ch ^ 48);
        ch = getchar();
    }
    k = x * f;
}
template<class T> inline void write(T x) {
    if (x < 0) {
        putchar('-'), x = -x;
    }
    if (x > 9) {
        write(x / 10);
    }
    putchar(x % 10 + '0');
}
i128 exgcd(i128 a, i128 b, i128 &x, i128 &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }

    i128 d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
template <typename T, typename U>
T ceil(T x, U y) {
  return (x > 0 ? (x + y - 1) / y : x / y);
}
template <typename T, typename U>
T floor(T x, U y) {
  return (x > 0 ? x / y : (x - y + 1) / y);
}
void solve() {
	i128 n, a, b, c, x;
    read(n);read(a);read(b);read(c);read(x);
    i128 ans = 0;
	for (i128 i = 1; i <= n; i ++) {
		//b * j + c * k = x - a * i;
		i128 j, k;
		i128 g = exgcd(b, c, j, k);
		i128 m = x - a * i;
		if (m % g) continue;
        
		j *= m / g, k *= m / g;
		
		i128 bb = b / g;
		i128 cc = c / g;
		//t是通解的常数
		//(1 - j) / c <= t <= (n - j) / c
		i128 l1 = ceil(1 - j, cc);
		i128 r1 = floor(n - j, cc);
		
		//k - n <= bt <= k - 1
		i128 l2 = ceil(k - n, bb);
		i128 r2 = floor(k - 1, bb);
		
		i128 l = max(l1, l2);
		i128 r = min(r1, r2);
		
		if (r >= l) ans += r - l + 1;
	}
	write(ans);
}
signed main(){
    // ios::sync_with_stdio(false);
    // cin.tie(nullptr);

    int t; t = 1;
    //cin >> t;
    
    while (t --) {
        solve();
    }

    return 0;
}

```



### CRT(中国剩余定理)

``` java
int exgcd(int a, int b, int &x, int &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }

    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

void merge(int &a, int &b, int c, int d) {
    if (a == -1 && b == -1) return ;
    int x , y;
    int g = exgcd(b , d , x , y);
    if ((c - a) % g) {
        a = b = -1;
    }
    d /= g;
    int t0 = ((c - a) / g) % d * x % d;
    if (t0 < 0) t0 += d;
    a = b * t0 + a;
    b = b * d;
    a %= b; //注意取模，否则大数会溢出
}
```



### 整除分块

**基本形式**

​																	$\Large\sum_{i=1}^{n}\lfloor \frac{n}{i}\rfloor$

考虑分块

$左端点为l,右端点为\large\lfloor \cfrac{n}{\lfloor\cfrac{n}{l}\rfloor}\rfloor$

**时间复杂度**

$O(\sqrt{n})$

```c++
//基本板子
for (int l = 1, r; l <= n; l = r + 1) {
    int o = k / l;
    if (o == 0) {
        r = n;
    } else {
    	r = min(n / o, n);
    }
}
```



### 欧拉函数

#### 求单个数的欧拉函数

$$\large O(\sqrt{n})$$

``` c++
int phi(int n) {
    int res = n;
    for (int i = 2; i * i  <= n; i ++) {
        if (n % i == 0) {
            res = res / i * (i - 1); //先除后乘
        }
        while (n % i == 0) {
            n /= i;
        }
	}
    if (n > 1) {
        res = res / n * (n - 1);
    }
    return res;
}
```

#### 求多个数的欧拉函数

$$\large O(n)$$

```c++
/*----------------------------------phi----------------------------------*/
int n, cnt, primes[N], phi[N];
int s[N];
bool st[N];
void sieve(int n) {
    phi[1] = 1;

    for (int i = 2; i <= n; i ++) {
        if (!st[i]) {
            primes[++cnt] = i;
            phi[i] = i - 1;
        }

        for (int j = 1; j <= cnt && i * primes[j] <= n; j ++) {
            st[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                phi[i * primes[j]] = phi[i] * primes[j];
                break;
            }
            phi[i * primes[j]] = phi[i] * (primes[j] - 1);
        }
    }

    for (int i = 1; i <= n; i ++) s[i] = s[i - 1] + phi[i]; //求前缀和
}
```

### 狄利克雷卷积

$\large 定义两个函数f,g的狄利克雷卷积为f*g,其自成一个函数:$

​									$\large (f*g)(n)=\sum_{d|n}f(d)g(\frac{n}{d})$

**常见的数论函数**

$\large \varepsilon(n)=\left\{\begin{matrix} 
  1 \qquad n=1  \\  
  0 \qquad otherwise
\end{matrix}\right.$

$\large I(n)=1$

$\large ID(n) = n$

$\large \varphi (n)=\sum_{i=1}^{n}[gcd(i,n)=1]$

$\large \mu(n)=\begin{equation}
\left\{
             \begin{array}{lr}
             1  \qquad n=1  \\
             (-1)_{r} \qquad n=p_{1}*p_{2}*p_{3}...p_{r}\\
             0 \qquad else &
             \end{array}
\right.
\end{equation}$



**常用的迪利克雷卷积**

​						$\large \mu*I=\epsilon$

​						$\large \varphi*I=id$

​						$\large \mu*id=\varphi$

​						$\large f*\epsilon=f$



### 莫比乌斯函数

**定义**

​						$\large \mu(n)=\begin{equation}
\left\{
​             \begin{array}{lr}
​             1  \qquad n=1  \\
​             (-1)^{r} \qquad n=p_{1}*p_{2}*p_{3}...p_{r}\\
​             0 \qquad else &
​             \end{array}
\right.
\end{equation}$



### 逆元

**0没有逆元**

#### 逆元存在的充分必要条件

$$\large gcd(a , p) = 1$$

#### 快速幂求逆元

$$\large O(\log(n))$$

``` c++
namespace inv {
  	int qpow(int a, int b, int p) {
    	int res = 1;
    	while (b) {
        	if (b & 1) res = res * a % p;
        	b >>= 1;
        	a = a * a % p;
    	}
    	return res;
	}
    
    int inv(int a, int p) {
        return qpow(a , p - 2 , p);
    }
     
};
```

#### 求1 - n每个数的逆元

$$O(n)$$

``` c++
int inv[N];
void get_inv() {
    inv[1] = 1;
    for (int i = 2; i <= N; i ++) {
        inv[i] = p - p / i * inv[p % i] % p;
    }
}
```



### 多项式

#### 快速傅里叶变换（FFT）

```c++
/*----------------------------------FFT----------------------------------*/
template<class T = double> struct __Complex {
    T x, y;
    __Complex() = default;
    __Complex(const T x, const T y) : x(x), y(y) {}
    __Complex &operator+=(const __Complex &b) {
        x += b.x;
        y += b.y;
        return *this;
    }
    __Complex &operator-=(const __Complex &b) {
        x -= b.x;
        y -= b.y;
        return *this;
    }
    __Complex &operator*=(const __Complex &b) {
        __Complex temp;
        temp.x = x * b.x - y * b.y;
        temp.y = x * b.y + y * b.x;
        *this = temp;
        return *this;
    }
    __Complex &operator*=(const double &b) {
        this -> x *= b;
        this -> y *= b;
        return *this;
    }
    __Complex &operator/=(const __Complex &b) {
        __Complex temp;
        temp.x = (x * b.x + y * b.y) / (b.x * b.x + b.y * b.y);
        temp.y = (y * b.x - x * b.y) / (b.x * b.x + b.y * b.y);
        *this = temp;
        return this;
    }
    __Complex &operator/=(const double b) {
        this -> x /= b;
        this -> y /= b;
        return *this;
    }
    __Complex operator+(const __Complex &b) {
        __Complex a = *this;
        a += b;
        return a;
    }
    __Complex operator-(const __Complex &b) {
        __Complex a = *this;
        a -= b;
        return a;
    }
    __Complex operator*(const __Complex &b) {
        __Complex a = *this;
        a *= b;
        return a;
    }
    friend ostream &operator<<(ostream &os, const __Complex &a) {
        os << a.x << " " << a.y;
        return os;
    }
};
using Complex = __Complex<>;
const long double PI = acos(-1.0);
const long double PI2 = PI / 2;
vector<Complex> r;
int preLg;
void pre(const int lg) {
    r.resize(1 << lg);
    for (int i = preLg ; i < lg ; i++) {
        int L = 1 << i;
        r[L] = Complex(cos(PI2 / L), sin(PI2 / L));
        for (int j = L + 1 ; j < (L << 1) ; j++) {
            r[j] = r[j - L] * r[L];
        }
    }
}
struct Poly {
    vector<Complex> a;
    Poly(const int size) {
        a.resize(size);
    }
    Complex &operator[](const int x) {
        return a[x];
    }
    void resize(const int n) {
        a.resize(n);
    }
    int size() {
        return a.size();
    }
    void FFT() {
        int n = a.size();
        for (int i = n ; i >= 2 ; i >>= 1) {
            int L = i >> 1;
            for (int j = 0 ; j != L ; j++) {
                Complex x = a[j], y = a[j + L];
                a[j] = x + y;
                a[j + L] = x - y;
            }
            for (int j = i, m = 1 ; j != n ; j += i, m++) {
                Complex rt = r[m];
                for (int k = 0 ; k != L ; k++) {
                    Complex x = a[j + k], y = a[j + k + L] * rt;
                    a[j + k] = x + y;
                    a[j + k + L] = x - y;
                }
            }
        }
    }
    void IFFT() {
        int n = a.size();
        for (int i = 2 ; i <= n ; i <<= 1) {
            int L = i >> 1;
            for (int j = 0 ; j != L ; j++) {
                Complex x = a[j], y = a[j + L];
                a[j] = x + y;
                a[j + L] = x - y;
            }
            for (int j = i, m = 1 ; j != n ; j += i, m++) {
                Complex rt = r[m];
                for (int k = 0 ; k != L ; k++) {
                    Complex x = a[j + k], y = a[j + k + L];
                    a[j + k] = x + y;
                    a[j + k + L] = (x - y) * rt;
                }
            }
        }
        double inv = 1.0 / n;
        for (int i = 0 ; i < n ; i++) {
            a[i] *= inv;
        }
        reverse(begin(a) + 1, end(a));
    }
    void mul(Poly &x) {
        Poly z(x);
        int n = 1, lg = 0, len = a.size() + x.size() - 1;
        while (n < len) {
            n <<= 1;
            lg++;
        }
        if (lg > preLg) {
            pre(lg);
            preLg = lg;
        }
        a.resize(n);
        z.resize(n);
        FFT();
        z.FFT();
        for (int i = 0 ; i < n ; i++) {
            a[i] *= z[i];
        }
        IFFT();
        a.resize(len);
    }
    void mulMe() {
        int n = 1, lg = 0, len = 2 * a.size() - 1;
        while (n < len) {
            n <<= 1;
            lg++;
        }
        if (lg > preLg) {
            pre(lg);
            preLg = lg;
        }
        a.resize(n);
        FFT();
        for (int i = 0 ; i < n ; i++) {
            a[i] *= a[i];
        }
        IFFT();
        a.resize(n - 1);
    }
}; // Poly
```

#### 快速数论变换（NTT）

```c++
const int N = 1e6+10;
const int p = 998244353, gg = 3, img = 332738118;
const int mod = 998244353;

int qpow(int a, int b) {
    int res = 1;
    while (b) {
        if (b & 1) res = 1ll * res * a % mod;
        a = 1ll * a * a % mod;
        b >>= 1;
    }
    return res;
}
/*----------------------------------Poly----------------------------------*/
namespace Poly {
    #define mul(x, y) (1ll * x * y >= mod ? 1ll * x * y % mod : 1ll * x * y)
    #define minus(x, y) (1ll * x - y < 0 ? 1ll * x - y + mod : 1ll * x - y)
    #define plus(x, y) (1ll * x + y >= mod ? 1ll * x + y - mod : 1ll * x + y)
    #define ck(x) (x >= mod ? x - mod : x)
 
    typedef vector<int> poly;
    const int G = 3;//根据具体的模数而定，原根可不一定不一样！！！
    //一般模数的原根为 2 3 5 7 10 6
    const int inv_G = qpow(G, mod - 2);
    int RR[N], deer[2][19][N], inv[N]; //此处的大小要比t大1
 
    // 预处理出来NTT里需要的w和wn，砍掉了一个log的时间
    // t为第一个大于多项式长度的2的正整数次方
    void init(const int t) {
        for (int p = 1; p <= t; p++) {
            int buf1 = qpow(G, (mod - 1) / (1 << p));
            int buf0 = qpow(inv_G, (mod - 1) / (1 << p));
            deer[0][p][0] = deer[1][p][0] = 1;
            for(int i = 1; i < (1 << p); ++ i) {
                deer[0][p][i] = 1ll * deer[0][p][i - 1] * buf0 % mod;//逆
                deer[1][p][i] = 1ll * deer[1][p][i - 1] * buf1 % mod;
            }
        }
        inv[1] = 1;
        for(int i = 2; i <= (1 << t); ++ i)
            inv[i] = 1ll * inv[mod % i] * (mod - mod / i) % mod;
    }
 
    int NTT_init(int n) {//快速数论变换预处理
        int limit = 1, L = 0;
        while(limit <= n) limit <<= 1, L ++ ;
        for(int i = 0; i < limit; ++ i)
            RR[i] = (RR[i >> 1] >> 1) | ((i & 1) << (L - 1));
        return limit;
    }
 
    void NTT(poly &A, int type, int limit) {//快速数论变换
        A.resize(limit);
        for(int i = 0; i < limit; ++ i)
            if(i < RR[i])
                swap(A[i], A[RR[i]]);
        for(int mid = 2, j = 1; mid <= limit; mid <<= 1, ++ j) {
            int len = mid >> 1;
            for(int pos = 0; pos < limit; pos += mid) {
                int *wn = deer[type][j];
                for(int i = pos; i < pos + len; ++ i, ++ wn) {
                    int tmp = 1ll * (*wn) * A[i + len] % mod;
                    A[i + len] = ck(A[i] - tmp + mod);
                    A[i] = ck(A[i] + tmp);
                }
            }
        }
        if(type == 0) {
            for(int i = 0; i < limit; ++ i)
                A[i] = 1ll * A[i] * inv[limit] % mod;
        }
    }
 
    poly poly_mul(poly A, poly B) {//多项式乘法
        int deg = A.size() + B.size() - 1;
        int limit = NTT_init(deg);
        poly C(limit);
        NTT(A, 1, limit);
        NTT(B, 1, limit);
        for(int i = 0; i < limit; ++ i)
            C[i] = 1ll * A[i] * B[i] % mod;
        NTT(C, 0, limit);
        C.resize(deg);
        return C;
    }
 
    poly poly_inv(poly &f, int deg) {//多项式求逆
        if(deg == 1)
            return poly(1, qpow(f[0], mod - 2));
 
        poly A(f.begin(), f.begin() + deg);
        poly B = poly_inv(f, (deg + 1) >> 1);
        int limit = NTT_init(deg << 1);
        NTT(A, 1, limit), NTT(B, 1, limit);
        for(int i = 0; i < limit; ++ i)
            A[i] = B[i] * (2 - 1ll * A[i] * B[i] % mod + mod) % mod;
        NTT(A, 0, limit);
        A.resize(deg);
        return A;
    }
 
    poly poly_dev(poly f) {//多项式求导
        int n = f.size();
        for(int i = 1; i < n; ++ i) f[i - 1] = 1ll * f[i] * i % mod;
        return f.resize(n - 1), f;//f[0] = 0，这里直接扔了,从1开始
    }
 
    poly poly_idev(poly f) {//多项式求积分
        int n = f.size();
        for(int i = n - 1; i ; -- i) f[i] = 1ll * f[i - 1] * inv[i] % mod;
        return f[0] = 0, f;
    }
 
    poly poly_ln(poly f, int deg) {//多项式求对数
        poly A = poly_idev(poly_mul(poly_dev(f), poly_inv(f, deg)));
        return A.resize(deg), A;
    }
 
    poly poly_exp(poly &f, int deg) {//多项式求指数
        if(deg == 1)
            return poly(1, 1);
 
        poly B = poly_exp(f, (deg + 1) >> 1);
        B.resize(deg);
        poly lnB = poly_ln(B, deg);
        for(int i = 0; i < deg; ++ i)
            lnB[i] = ck(f[i] - lnB[i] + mod);
 
        int limit = NTT_init(deg << 1);//n -> n^2
        NTT(B, 1, limit), NTT(lnB, 1, limit);
        for(int i = 0; i < limit; ++ i)
            B[i] = 1ll * B[i] * (1 + lnB[i]) % mod;
        NTT(B, 0, limit);
        B.resize(deg);
        return B;
    }
 
    poly poly_sqrt(poly &f, int deg) {//多项式开方
        if(deg == 1) return poly(1, 1);
        poly A(f.begin(), f.begin() + deg);
        poly B = poly_sqrt(f, (deg + 1) >> 1);
        poly IB = poly_inv(B, deg);
        int limit = NTT_init(deg << 1);
        NTT(A, 1, limit), NTT(IB, 1, limit);
        for(int i = 0; i < limit; ++ i)
            A[i] = 1ll * A[i] * IB[i] % mod;
        NTT(A, 0, limit);
        for(int i =0; i < deg; ++ i)
            A[i] = 1ll * (A[i] + B[i]) * inv[2] % mod;
        A.resize(deg);
        return A;
    }
 
    poly poly_pow(poly f, int k) {//多项式快速幂
        f = poly_ln(f, f.size());
        for(auto &x : f) x = 1ll * x * k % mod;
        return poly_exp(f, f.size());
    }
 
    poly poly_cos(poly f, int deg) {//多项式三角函数（cos）
        poly A(f.begin(), f.begin() + deg);
        poly B(deg), C(deg);
        for(int i = 0; i < deg; ++ i)
            A[i] = 1ll * A[i] * img % mod;
 
        B = poly_exp(A, deg);
        C = poly_inv(B, deg);
        int inv2 = qpow(2, mod - 2);
        for(int i = 0; i < deg; ++ i)
            A[i] = 1ll * (1ll * B[i] + C[i]) % mod * inv2 % mod;
        return A;
    }
 
    poly poly_sin(poly f, int deg) {//多项式三角函数（sin）
        poly A(f.begin(), f.begin() + deg);
        poly B(deg), C(deg);
        for(int i = 0; i < deg; ++ i)
            A[i] = 1ll * A[i] * img % mod;
 
        B = poly_exp(A, deg);
        C = poly_inv(B, deg);
        int inv2i = qpow(img << 1, mod - 2);
        for(int i = 0; i < deg; ++ i)
            A[i] = 1ll * (1ll * B[i] - C[i] + mod) % mod * inv2i % mod;
        return A;
    }
 
    poly poly_arcsin(poly f, int deg) {
        poly A(f.size()), B(f.size()), C(f.size());
        A = poly_dev(f);
        B = poly_mul(f, f);
        for(int i = 0; i < deg; ++ i)
            B[i] = minus(mod, B[i]);
        B[0] = plus(B[0], 1);
        C = poly_sqrt(B, deg);
        C = poly_inv(C, deg);
        C = poly_mul(A, C);
        C = poly_idev(C);
        return C;
    }
 
    poly poly_arctan(poly f, int deg) {
        poly A(f.size()), B(f.size()), C(f.size());
        A = poly_dev(f);
        B = poly_mul(f, f);
        B[0] = plus(B[0], 1);
        C = poly_inv(B, deg);
        C = poly_mul(A, C);
        C = poly_idev(C);
        return C;
    }
}//Poly
```

#### 大质数模数的应用

另外，在做题过程中，如果需要用到多项式卷积的地方，但是答案非常大，超过了int但是在long long之内，FFT的精度往往不够

我们可以考虑使用$10^{18}$级别的大质数来当做NTT的模数，这样就可以让结果不被取模掉，同时又可以起到卷积加速的作用

需要注意的是中间结果仍然可能爆long long，所以中间结果（乘法时）可能需要转int128，然后再取模，最常用的大质数有：

- $p = 1945555039024054273 = 27 *2^{56} + 1,g=5$
- $p = 4179340454199820289 = 29 *2^{57} + 1,g=3$

## 博弈论

### SG函数

$sg(x):= mex(sg(y)|x\to y)$

这里的$x, y$都是表示某种状态

$\left\{\begin{matrix} 
   sg = 0 \Rightarrow 不存在后继等于0 \Rightarrow 必败\\  
  sg > 0 \Rightarrow 存在后继等于0\Rightarrow 必胜
\end{matrix}\right.$

$sg定理:sg(G) = sg(G_{1}) \oplus sg(G_{2}) \oplus sg(G_{3})...sg(G_{n})$



### 阶梯Nim

**结论：偶数堆石子的异或和**



### 树上博弈

#### Colon Principle（克朗原理）

$\large 首先，我们约定对任意一个节点的SG值是以这个点为根张成的子树的SG值，对于某一个点x,$

$\large 设其子节点为x_{1},x_{2},x_{3}...x_{t}$

​		$\large SG(x) = 0, t=0$

​		$\large SG(x)=(SG(x_{1})+1)\oplus(SG(x_{2})+1))\oplus(SG(x_{3})+1)...\oplus(SG(x_{n})+1)$

## 组合数学

### 结论

#### 若在$1$到$n$中，两个相邻数字至少要选一个，一共有多少种方案？

考虑令$n$个数的方案为$F(n)$。分类讨论

+ 若选$n$，则方案数为$F(n - 1)$

* 若不选$n$，则必须选$n-1$，方案数为$F(n - 2)$
* 撒大苏打

显然$F(1)=1,F(2)=3,F(n)=F(n-1)+F(n-2)$

### 求组合数

``` c ++
int fact[N], infact[N];
int qpow(int a , int b , int p){
    int res = 1;
    while(b){
        if(b & 1) res = (i64)res * a % p;
        b >>= 1;
        a = (i64)a * a % p;
    }
    return res % p;
}

void init(){
    fact[0] = 1;
    infact[0] = 1;
    for(int i = 1; i < N; i++){
        fact[i] = (LL)fact[i - 1] * i % p;
        infact[i] = (LL)infact[i - 1] * qpow(i , p - 2 , p) % p;
    }
}
int C(int a , int b){
    return fact[a] * infact[b] % p * infact[a - b] % p;
}
```

#### 求大组合数

```c++
namespace Lucas {
    const int mod = 1000000007;
    long long inv(long long a) {
        long long res = 1, b = mod - 2;
        while (b) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }
    long long comb(const long long& n, const long long& m) {
        long long x = 1, y = 1;
        if (n < m) return 0;
        else if (n == m) return 1;
        for (int i = n - m + 1 ; i <= n ; i++) x = x * i % mod;
        for (int i = 1 ; i <= m ; i++) y = y * i % mod;
        return x * inv(y) % mod;
    }
    long long get(const long long& n, const long long& m) {
      if (m == 0) return 1;
      return (comb(n % mod, m % mod) * get(n / mod, m / mod)) % mod;
    }
} // Lucas
```

### 线性基

**性质**

- 原数列里的任何一个数都可以通过线性基里的数异或表示出来
- 线性基里任意一个子集的异或和都不为0
- 一个数列可能有多个线性基，但是线性基里数的数量一定唯一，而且是满足性质一的基础上最少的

```c ++
i64 p[100], tmp[100]; //存储线性基
int cnt, zero;
//插入
void insert(i64 x) {
    for (int i = 62; i >= 0; i --) {
        if (!(x >> i)) continue ;
        if (!p[i]) {
            p[i] = x;
            return;
        }
        x ^= p[i];
    }
    zero = true;
}
//询问是否能被异或出来
bool ask_exist(i64 x) {
    for (int i = 62; i >= 0; i --) {
        if (x >> i & 1) x ^= p[i];
    }
    return x == 0;
}
//询问最小值
i64 ask_mn() {
    if (zero) return 0;
    for (int i = 0; i <= 62; i ++) {
        if (p[i]) return p[i];
    }
}
//询问最大值
i64 ask_mx() {
    i64 ans = 0;
    for (int i = 62; i >= 0; i --) {
        ans = max(ans, ans ^ p[i]);
    }
    return ans;
}
void rebuild() {
    for (int i = 0; i <= 63; i ++)
        for (int  j = i - 1; j >= 0; j --)
            if (p[i] >> j & 1)
                p[i] ^= p[j];

    for (int i = 0; i <= 63; i ++)
        if (p[i])
            d[cnt ++] = p[i];
}

//if (zero) x --
i64 ask_kth(i64 x) {
    if (x >= (1ull << cnt))
        return -1;
    i64 res = 0;
    for (int i = 0; i <= 63; i ++)
        if (x >> i & 1)
            res ^= tmp[i];

    return res;
}
```

#### 2023桂林站C Master of Both IV(线性基，数论)

**题意：**

​		给一个可重集，求有多少子集满足每个元素都可以被异或和整除。

**思路**

​		易得异或值要么是$0$，要么是$max(s)$

* 当异或和为$0$时，相当于求多少子序列异或和为$0$，设这个集合的异或线性基的秩为$r$，则答案为${2}^{n - r}$
* 当异或和为$max(s)$，枚举最大值，然后枚举所有最大值的因子加入线性基即可

$时间复杂度O(n{log}^{2}n)$

**代码**

```c++
#include<bits/stdc++.h>
#define x first
#define y second
#define endl '\n'

using namespace std;
using i64 = int64_t;

const int N = 200010;
const int p = 998244353;
struct linear_basis {
    int num[50];
    int rank; //线性基的秩
    void init() {
        for (int i = 0; i < 31; i ++) {
            num[i] = 0;
        }
        rank = 0;
    }
    bool insert(int x) {
        for (int i = 30; i >= 0; i --) {
            if (x >> i & 1) {
                if (!num[i]) {
                    num[i] = x;
                    ++ rank;
                    return true;
                } else {
                    x ^= num[i];
                }
            }
        }
        return false;
    }
}line[N];
int pw[N];
void solve() {
    int n;
    cin >> n;

    for (int i = 0; i <= n; i ++) {
        line[i].init();
    }
    unordered_map<int, int> cnt, mp;
    for (int i = 1; i <= n; i ++) {
        int x;
        cin >> x;
        cnt[x] ++; 
    }
    int res = 0;
    for (int i = 1; i <= n; i ++) {
        if (!cnt[i]) continue;
        for (int j = 0; j * i <= n; j ++) {
            mp[j * i] += cnt[i];
            line[j * i].insert(i);
        }
    }

    res += (pw[n - line[0].rank] - 1 + p) % p;
    for (int i = 1; i <= n; i ++) {
        if (!cnt[i]) continue;
        res += pw[mp[i] - line[i].rank];
        res %= p;
    }
    cout << res << endl;
}
signed main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t; t = 1;
    cin >> t;
    pw[0] = 1;
    for (int i = 1; i < N; i ++) {
        pw[i] = pw[i - 1] * 2 % p;
    }
    while (t --) {
        solve();
    }

    return 0;
}
```



### 容斥原理



**例题:求出第k个与n互质的数**

思路:

首先对n进行质因数分解，然后容斥求出[1,mid]中与n互质的数有多少个

```c++
// Problem: 第K小互质数
// Contest: NowCoder
// URL: https://ac.nowcoder.com/acm/contest/60254/J
// Memory Limit: 524288 MB
// Time Limit: 2000 ms

#include<bits/stdc++.h>
#define int long long
#define x first
#define y second
#define PII pair <int, int>
#define endl '\n'
const int INF = 0x3f3f3f3f;

using namespace std;
using i64 = long long;
i64 mul(i64 a, i64 b, i64 m) {
    return static_cast<__int128>(a) * b % m;
}
i64 power(i64 a, i64 b, i64 m) {
    i64 res = 1 % m;
    for (; b; b >>= 1, a = mul(a, a, m))
        if (b & 1)
            res = mul(res, a, m);
    return res;
}
bool isprime(i64 n) {
    if (n < 2)
        return false;
    static constexpr int A[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    int s = __builtin_ctzll(n - 1);
    i64 d = (n - 1) >> s;
    for (auto a : A) {
        if (a == n)
            return true;
        i64 x = power(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool ok = false;
        for (int i = 0; i < s - 1; ++i) {
            x = mul(x, x, n);
            if (x == n - 1) {
                ok = true;
                break;
            }
        }
        if (!ok)
            return false;
    }
    return true;
}
std::vector<i64> factorize(i64 n) {
    std::vector<i64> p;
    std::function<void(i64)> f = [&](i64 n) {
        if (n <= 10000) {
            for (int i = 2; i * i <= n; ++i)
                for (; n % i == 0; n /= i)
                    p.push_back(i);
            if (n > 1)
                p.push_back(n);
            return;
        }
        if (isprime(n)) {
            p.push_back(n);
            return;
        }
        auto g = [&](i64 x) {
            return (mul(x, x, n) + 1) % n;
        };
        i64 x0 = 2;
        while (true) {
            i64 x = x0;
            i64 y = x0;
            i64 d = 1;
            i64 power = 1, lam = 0;
            i64 v = 1;
            while (d == 1) {
                y = g(y);
                ++lam;
                v = mul(v, std::abs(x - y), n);
                if (lam % 127 == 0) {
                    d = std::gcd(v, n);
                    v = 1;
                }
                if (power == lam) {
                    x = y;
                    power *= 2;
                    lam = 0;
                    d = std::gcd(v, n);
                    v = 1;
                }
            }
            if (d != n) {
                f(d);
                f(n / d);
                return;
            }
            ++x0;
        }
    };
    f(n);
    std::sort(p.begin(), p.end());
    return p;
}
void solve() {
	int n, k;
	cin >> n >> k;
	
	set<int> s;
	vector<int> a = factorize(n);

	for (auto c : a) s.insert(c);
	vector<int> p(s.begin(), s.end());
	
	int l = 1, r = n;
	while (l < r) {
		int mid = l + r >> 1;
		
		int sum = 0;
		
		function<void(int, int, int)> dfs = [&](int i, int s, bool c) {
			if (i == p.size()) {
				if (c) {
					sum += s;
				} else {
					sum -= s;
				}
				return ;
			}
			dfs(i + 1, s, c);
			dfs(i + 1, s / p[i], not c);
		};
		
		dfs(0, mid, true);
		if (sum >= k) {
			r = mid;
		} else {
			l = mid + 1;
		}
	}
	
	cout << r << endl;
}
signed main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int t; t = 1;
    //cin >> t;
    
    while (t --) {
        solve();
    }

    return 0;
}

```

## 字符串

### Hash

```c++
struct stringHash {
    static const int BASE1 = 29, MOD1 = 1e9 + 7;
    static const int BASE2 = 131, MOD2 = 1e9 + 9;

    int n;  // 串长
    string s;  // 下标从 1 开始
    vector<int> ha1, ha2;  // 正着的哈希值
    vector<int> pow1, pow2;  // BASE1, BASE2 的乘方

    stringHash() {}

    stringHash(string _s) : s(' ' + _s), n(_s.length()) {  // _s 下标从 0 开始
        ha1 = vector<int>(n + 5), ha2 = vector<int>(n + 5);
        pow1 = vector<int>(n + 5, 1), pow2 = vector<int>(n + 5, 1);

        for (int i = 1; i <= n; i++) {
            pow1[i] = (ll)pow1[i - 1] * BASE1 % MOD1;
            pow2[i] = (ll)pow2[i - 1] * BASE2 % MOD2;
        }

        init();
    }

    void init() {
        for (int i = 1; i <= n; i++) {
            ha1[i] = ((ll)ha1[i - 1] * BASE1 % MOD1 + s[i]) % MOD1;
            ha2[i] = ((ll)ha2[i - 1] * BASE2 % MOD2 + s[i]) % MOD2;
        }
    }

    pair<int, int> get(int l, int r) {  // 求 s[l ... r] 的哈希值
        int res1 = ((ha1[r] - (ll)ha1[l - 1] * pow1[r - l + 1] % MOD1) % MOD1 + MOD1) % MOD1, 
            res2 = ((ha2[r] - (ll)ha2[l - 1] * pow2[r - l + 1] % MOD2) % MOD2 + MOD2) % MOD2;
        return pair<int, int>(res1, res2);
    }
};

```

### 字典树（Trie)

```c++
struct Trie {
    const static int N = 100000;
    int tr[N][26], cnt[N], tree_size;
    void clear() {
        for (int i = 0 ; i < tree_size ; i++) {
            for (int j = 0 ; j < 26 ; j++) {
                tr[i][j] = 0;
            }
            cnt[i] = 0;
        }
        tree_size = 0;
    }
    void insert(const string& str) {
        int len = str.size(), index = 0;
        for (int i = 0 ; i < len ; i++) {
            const int c = str[i] - 'a';
            if (tr[index][c] == 0) tr[index][c] = ++tree_size;
            index = tr[index][c];
        }
        cnt[index]++;
    }
    int find(const string& str) {
        int len = str.size(), index = 0;
        for (int i = 0 ; i < len ; i++) {
            const int c = str[i] - 'a';
            if (tr[index][c] == 0) return 0;
            index = tr[index][c];
        }
        return cnt[index];
    }
}; // Trie
```



## 数据结构

### 线段树



### 树状数组

**树状数组的下标不能为0**

```c++
template<class T>
struct Fenwick {
    int n;
    vector<T> a;

    Fenwick(int n = 0) {
        init(n);
    }

    void init(int n) {
        this->n = n;
        a.assign(n, T());
    }

    void add(int p, T x) {
        for (int i = p; i < n; i += i & -i) {
            a[i] += x;
        }
    }

    T sum(int p) {
        T res = 0;
        for (int i = p; i > 0; i -= i & -i) {
            res += a[i];
        }
        return res;
    }
};
```

#### 树状数组维护区间种类数

```c++
void solve() {
	int n;
	cin >> n;

	vector<int> a(n + 1); 
	for (int i = 1; i <= n; i ++) {
		cin >> a[i]; //i下标的种类
	}

	int m;
	cin >> m;
	vector<node> query;
	for (int i = 1; i <= m; i ++) {
		int l, r;
		cin >> l >> r;
		query.push_back({l, r, i}); //离线下来
	}

	sort(query.begin(), query.end(), [&](node a, node b) {
		return a.r < b.r; 
	});


	Fenwick<int> tr(1000010);
	vector<int> vis(1000010);
	int pos = 0;
	vector<int> ans(m + 1);
	for (auto [l, r, id] : query) {
		while (pos + 1 <= r) {
			pos ++;
			tr.add(pos, 1);
			if (vis[a[pos]]) {
				tr.add(vis[a[pos]], -1);
			}
			vis[a[pos]] = pos;
		}
		ans[id] = tr.sum(r) - tr.sum(l - 1);
	}
	for (int i = 1; i <= m; i ++) {
		cout << ans[i] << endl;
	}

}
```

### 分块

$$O(1)$$单点修改，$O(\sqrt n)$区间查询

**当修改次数多，查询次数少时利用分块**

```c ++
//M为块的数量
//len为每个块的长度(通常为sqrtn)
struct Blocks {
  int val[N], blk[M];
  
  void clear() {
    memset(val, 0, sizeof val);
    memset(blk, 0, sizeof blk);
  }
  
  void add(int x) {
    val[x]++;
    blk[x / len]++; 
  }
  
  int ask(int x) {
    if (!x) return 0;
    int res = 0;
    while ((x + len) % len != len - 1) res += val[x--];
    if (x < 0) return res;
    x /= len;
    while (x >= 0) res += blk[x--];
    return res;
  }
} b;

```

### ST表

**利用倍增思想**

$O(nlogn)$处理，$O(1)$查询区间最大/最小值

```c++
template<class T> struct SparseTable {
    vector<vector<T>> st;
    vector<int> lg;
    SparseTable(const vector<T>& s) {
        const int N = s.size();
        st.assign(N, vector<T>(22, 0));
        lg.resize(N);
        for (int i = 2 ; i < N ; i++) lg[i] = lg[i >> 1] + 1;
        for (int i = 1 ; i < N ; ++i) st[i][0] = s[i];
        for (int j = 1 ; j <= lg[N - 1] ; j++) {
            for (int i = 1 ; i + (1 << j) - 1 < N ; i++) {
                st[i][j] = min(st[i][j - 1], st[i + (1 << j - 1)][j - 1]);
            }
        }
    }
    T query(int L, int R) {
        int k = lg[R - L + 1];
        return min(st[L][k], st[R - (1 << k) + 1][k]);
    }
}; // SparseTable
```

## 图论

### 强联通分量

>  在**有向图**G中，如果两个顶点u，ｖ间有一条从ｕ到ｖ的有向路径，同时还有一条从ｖ到ｕ的有向路径，则称**两个顶点强连通**。如果有向图G的每两个顶点都强连通，称G是一个强连通图。有向非强连通图的极大强连通子图，称为**强连通分量**。

### 割点

> 对于一个无向图，如果把一个点删除后这个图的极大连通分量数增加了，那么这个点就是这个图的割点（又称割顶）。

### 割边

和割点差不多，叫做桥。

>  对于一个无向图，如果删掉一条边后图中的连通分量数增加了，则称这条边为桥或者割边。
>
> 

### 无向图三元环计数

**题意：**

​		给定$n$个点$m$条边的简单无向图,求其三元环的数量

​		$m$表示边的数量

​		时间复杂度$O(m\sqrt m)$

```c++
#include<bits/stdc++.h>
#define x first
#define y second
#define endl '\n'

using namespace std;
using i64 = int64_t;

vector<int> g[200010];
void solve() {
    int n, m;
    cin >> n >> m;

    vector<int> a(m + 1), b(m + 1);
    vector<int> deg(n + 1);
    for (int i = 1; i <= m; i ++) {
        cin >> a[i] >> b[i];
        deg[a[i]] ++;
        deg[b[i]] ++;
    }   

    for (int i = 1; i <= m; i ++) {
        int u = a[i];
        int v = b[i];
        if (deg[u] > deg[v]) {
            swap(u, v);
        } else if (deg[u] == deg[v] && u > v) {
            swap(u, v);
        }
        g[u].push_back(v);
    }

    vector<int> vis(n + 1);
    int ans = 0;
    for (int u = 1; u <= n; u ++) {
        for (auto v: g[u]) {
            vis[v] = u;
        }
        for (auto v : g[u]) {
            for (auto w : g[v]) {
                if (vis[w] == u) {
                    ans ++;
                }
            }
        }
    }
    cout << ans << endl;    
}
signed main(){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t; t = 1;
    //cin >> t;

    while (t --) {
        solve();
    }

    return 0;
}
```

### 网络流

#### dinic

```c++
using LL = long long;
const int maxn = 1e5 + 5, maxm = 2e6 + 5;
template<typename flow_t>
struct MaxFlow{
 
    const flow_t INF = numeric_limits<flow_t>::max() / 2;
 
    int h[maxn], e[maxm], ne[maxm], idx;
    flow_t f[maxm];
    int cur[maxn], q[maxn], d[maxn];
    int V, S, T;
 
    void init(int v, int s, int t){
        for(int i = 0; i <= v; i++) h[i] = -1;
        idx = 0;
        V = v, S = s, T = t;
    }
    
    void add(int a, int b, flow_t c, flow_t d = 0){
        e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx++;
        e[idx] = a, f[idx] = d, ne[idx] = h[b], h[b] = idx++;
    }
    
    bool bfs(){
        for(int i = 0; i <= V; i++) d[i] = -1;
        int hh = 0, tt = -1;
        q[++tt] = S, d[S] = 0, cur[S] = h[S];
        while(hh <= tt){
            int t = q[hh++];
            for(int i = h[t]; ~i; i = ne[i]){
                int j = e[i];
                if (d[j] == -1 && f[i]){
                    d[j] = d[t] + 1;
                    cur[j] = h[j];
                    if (j == T) return true;
                    q[++tt] = j;
                }
            }
        }
        return false;
    }
    
    flow_t find(int u, flow_t limit){
        if (u == T) return limit;
        flow_t flow = 0;
        // start from cur[u] instead of h[u] <- important
        for(int i = cur[u]; ~i && flow < limit; i = ne[i]){
            int j = e[i];
            cur[u] = i;
            if (d[j] == d[u] + 1 && f[i]){
                flow_t t = find(j, min(f[i], limit - flow));
                if (!t) d[j] = -1;
                else f[i] -= t, f[i ^ 1] += t, flow += t; 
            }
        }
        return flow;
    }
    
    flow_t dinic(){
        flow_t res = 0, flow;
        while(bfs()) while(flow = find(S, INF)) res += flow;
        return res;
    }
};
 
MaxFlow<int> flow;

```

#### 最小费用流

```c++
using LL = long long;
const int maxn = 1e4 + 5, maxm = 1e6 + 5;
template<typename cost_t>
struct MinCostMaxFlow{
 
    const cost_t INF = numeric_limits<cost_t>::max() / 2;
    int h[maxn], e[maxm], ne[maxm], idx;
    cost_t f[maxm], w[maxm], d[maxn], incf[maxn];
    int q[maxn], pre[maxn];
    bool vis[maxn];
    int V, S, T;
 
    void init(int v, int s, int t){
        for(int i = 0; i <= v; i++) h[i] = -1;
        idx = 0;
        V = v, S = s, T = t;
    }
 
    void add(int a, int b, cost_t c, cost_t d){
        e[idx] = b, f[idx] = c, w[idx] = d, ne[idx] = h[a], h[a] = idx++;
        e[idx] = a, f[idx] = 0, w[idx] = -d, ne[idx] = h[b], h[b] = idx++;
    }
 
    bool spfa(){
        int hh = 0, tt = 0;
        for(int i = 0; i <= V; i++){
            d[i] = INF;
            incf[i] = 0;
            vis[i] = 0;
        }
        q[tt++] = S, d[S] = 0, incf[S] = INF;
        while(hh != tt){
            int t = q[hh++];
            if (hh == maxn) hh = 0;
            vis[t] = 0;
            for(int i = h[t]; ~i; i = ne[i]){
                int j = e[i];
                if (f[i] && d[j] > d[t] + w[i]){
                    d[j] = d[t] + w[i];
                    incf[j] = min(incf[t], f[i]);
                    pre[j] = i;
                    if (!vis[j]){
                        vis[j] = 1;
                        q[tt++] = j;
                        if (tt == maxn) tt = 0;
                    }
                }
            }
        }
        return incf[T] > 0;
    }
 
    pair<cost_t, cost_t> EK(){
        cost_t flow = 0, cost = 0;
        while(spfa()){
            cost_t t = incf[T];
            flow += t, cost += d[T] * t;
            for(int i = T; i != S; i = e[pre[i] ^ 1]){
                f[pre[i]] -= t, f[pre[i] ^ 1] += t;
            }
        }
        return {flow, cost};
    }
};
MinCostMaxFlow<int> flow;

```







