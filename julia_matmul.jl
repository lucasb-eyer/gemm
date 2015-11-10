const A = map(Float32, randn(10*1024, 10*1024));
const B = map(Float32, randn(10*1024, 10*1024));

A*B

tmin = Inf
for i=1:10
    tic()
    A*B
    tmin = min(tmin, toc())
end

print(tmin)
