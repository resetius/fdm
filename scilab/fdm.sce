
fmdinitialized=0;

function initfdm()
    if (fmdinitialized <> 1) then
        link("fdm", [
        "fdm_lapl1d",
        "fdm_lapl2d"
        ], "c")
        fmdinitialized = 1;
    end
endfunction

function f = fdm_lapl1d(x, l_x)
    n   = size(x, 1)
    n_x = n-1
    f = call("fdm_lapl1d", ...
        x, 2, "d",...
        l_x, 3, "d",...
        n_x, 4, "i",...
        "out",...
        [n,1],1,"d"...
    )
endfunction
