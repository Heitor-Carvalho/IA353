% Function with minimun at -1.5
func = @(x) x.^2 + 3*x + 2;
assert(golden_search(-2, 2, func, 0.01) < -1.49 && golden_search(-2, 2, func, 0.01) > -1.51)
