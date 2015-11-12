const N = 10
const A = map(Float32, randn(N*1024,N*1024));
const B = map(Float32, randn(N*1024,N*1024));

A*B

tmin = Inf
for i=1:10
    tic()
    A*B
    tmin = min(tmin, toc())
end

print(tmin)
