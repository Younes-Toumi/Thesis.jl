using PlotlyJS

N = 700
layout = Layout(
    scene=attr(
        xaxis=attr(
            nticks=4,
            range=[-100,100]
        ),
        yaxis=attr(
            nticks=4,
            range=[-50,100]
        ),
        zaxis=attr(
            nticks=4,
            range=[-100,100]
        ),
    ),
    width=700,
    margin=attr(
        r=20,
        l=10,
        b=10,
        t=10
    ),
)

plot(mesh3d(
        x=(70 .* randn(N)),
        y=(55 .* randn(N)),
        z=(10 .* randn(N)),
        color="rgba(244,22,100,0.6)"
    ),
    layout,
)