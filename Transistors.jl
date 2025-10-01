using LinearAlgebra
abstract type Transistor end
abstract type SingleChannelTransistor <: Transistor end

function Ber(z)
    if (z < 1e-8)
        return 1.0
    else 
        return z / (exp(z) - 1.0)
    end
end

function solveTridiagonal!(A::Tridiagonal, B::Vector{Float64}, x::AbstractArray)
    N = length(B)
    
    for i=2:N
        w = A[i,i-1] / A[i-1,i-1]
        A[i,i] = A[i,i] - w*A[i-1,i]
        B[i] = B[i] - w*B[i-1]
    end

    x[N] = B[N] / A[N,N]

    for i=N-1:-1:1
        x[i] = (B[i] - A[i,i+1]*x[i+1]) / A[i,i]
    end
end

function argumentMin(A::Matrix, n::Int)
    Nx = size(A)[2]
    minidx = 1
    minnum = A[n,1] 
    for i=1:Nx
        if (A[n, i] < minnum)
            minnum = A[n,i]
            minidx = i
        end
    end
    return minidx
end

struct Material
	mn :: Float64
	mp :: Float64
	D :: Float64
	T :: Float64
	EG :: Float64
	mu :: Float64
	eps :: Float64
	gn :: Int
	gp :: Int
    ballistic :: Bool
	NC :: Float64
	NV :: Float64
	NDOP :: Float64
	VT :: Float64
	
	function Material(mn :: Float64, mp :: Float64, D :: Float64, T :: Float64, EG :: Float64, mu :: Float64, eps:: Float64, gn :: Int, gp :: Int, ballistic=false)
		m0 = 9.1093837015e-31
		kb = 1.380649e-23
		e0 = 1.60217663e-19
		hbar = 1.054571817e-34
		NC = (gn*mn*m0*kb*T) / (π * hbar * hbar)
		NV = (gp*mn*m0*kb*T) / (π * hbar * hbar)
		VT = (kb * T) / (e0)
		NDOP = NC * log(1.0 + exp((-1.0 * D) / VT)) - NV * log(1.0 + exp((-EG + D) / VT))
		new(mn, mp, D, T, EG, mu, eps, gn, gp, ballistic, NC, NV, NDOP, VT)
	end
	
end

struct SingleGateTransistor <: SingleChannelTransistor
    Lch :: Float64
	tins :: Float64
	tsd :: Float64
    Lun :: Float64
    m :: Material
	dx :: Float64
	Nx :: Int
    Ng :: Int
    Nu :: Int
	Ny :: Int
	Nsd :: Int
	Nins :: Int
	phi :: Array{Float64, 2}
	phi_old :: Array{Float64, 2}
	n :: Vector{Float64}
    ddlhs :: Tridiagonal
    ddrhs :: Vector{Float64}
	
	function SingleGateTransistor(Lg :: Float64, tins :: Float64, tsd :: Float64, Lun::Float64, m::Material, dx :: Float64)
        Ng = round(Int, (Lg / dx)) + 1
		Nu = round(Int, (Lun / dx))
        Nx = 2*Nu + Ng
		Nins = round(Int, (tins / dx)) + 1
		Nsd = round(Int, (tsd / dx)) + 1
		Ny = Nins + Nsd + 1
		phi = zeros(Float64, Ny, Nx)
		phi_old = zeros(Float64, Ny, Nx)
		n = zeros(Float64, Nx)
        dd = zeros(Float64,Nx)
        dl = zeros(Float64, Nx-1)
        du = zeros(Float64, Nx-1)
        ddlhs = Tridiagonal(dl,dd,du)
        ddrhs = zeros(Float64, Nx)
		new(Lg, tins, tsd, Lun, m, dx, Nx, Ng, Nu, Ny, Nsd, Nins, phi, phi_old, n, ddlhs, ddrhs)
        
	end
end

struct  DoubleGateTransistor <: SingleChannelTransistor
	Lg :: Float64
	tins :: Float64
	tsd :: Float64
    Lun :: Float64
    m :: Material
	dx :: Float64
	Nx :: Int
    Ng :: Int
    Nu :: Int
	Ny :: Int
	Nsd :: Int
	Nins :: Int
	phi :: Array{Float64, 2}
	phi_old :: Array{Float64, 2}
	n :: Vector{Float64}
    ddlhs :: Tridiagonal
    ddrhs :: Vector{Float64}
	
	function DoubleGateTransistor(Lg :: Float64, tins :: Float64, tsd :: Float64, Lun::Float64, m:: Material, dx :: Float64)
        Ng = round(Int, (Lg / dx)) + 1
		Nu = round(Int, (Lun / dx))
        Nx = 2*Nu + Ng
		Nins = round(Int, (tins / dx)) + 1
		Nsd = round(Int, (tsd / dx)) + 1
		Ny = 2*Nins + Nsd + 1
		phi = zeros(Float64, Ny, Nx)
		phi_old = zeros(Float64, Ny, Nx)
		n = zeros(Float64, Nx)
        dd = zeros(Float64, Nx)
        du = zeros(Float64, Nx-1)
        dl = zeros(Float64, Nx-1)
        ddrhs = zeros(Float64, Nx)
        ddlhs = Tridiagonal(dl, dd, du)
		new(Lg, tins, tsd, Lun, m, dx, Nx, Ng, Nu, Ny, Nsd, Nins, phi, phi_old, n, ddlhs, ddrhs)
	end
end

struct DoubleChannelTransistor <: Transistor
	Lg :: Float64
	tins :: Float64
	tsd :: Float64
    Lun :: Float64
    m :: Material
	dx :: Float64
	Nx :: Int
    Ng :: Int
    Nu :: Int
	Ny :: Int
	Nsd :: Int
	Nins :: Int
	phi :: Array{Float64, 2}
	phi_old :: Array{Float64, 2}
	n :: Array{Float64, 2}
	ddlhs :: Tridiagonal
    ddrhs :: Vector{Float64}

	function DoubleChannelTransistor(Lg :: Float64, tins :: Float64, tsd :: Float64, Lun::Float64, m :: Material, dx :: Float64)
		Ng = round(Int, (Lg / dx)) + 1
		Nu = round(Int, (Lun / dx))
        Nx = 2*Nu + Ng
		Nins = round(Int, (tins / dx)) + 1
		Nsd = round(Int, (tsd / dx)) + 1
		Ny = 2*Nins + Nsd + 2
		phi = zeros(Float64, Ny, Nx)
		phi_old = zeros(Float64, Ny, Nx)
		n = zeros(Float64, Nx, 2)
        dd = zeros(Float64, Nx)
        dl = zeros(Float64, Nx-1)
        du = zeros(Float64, Nx-1)
        ddlhs = Tridiagonal(dl, dd, du)
        ddrhs = zeros(Float64, Nx)
		new(Lg, tins, tsd, Lun, m, dx, Nx, Ng, Nu, Ny, Nsd, Nins, phi, phi_old, n, ddlhs, ddrhs)
	end
end


function solvePoisson!(t::SingleGateTransistor, VG::Float64, VD::Float64, sor::Float64, tol::Float64, maxIter::Int)
    new_ = 0.0
	error = tol + 1.
	N = t.Nx * t.Ny
	k = (1.60217663e-19*t.dx / (t.m.eps * 8.85418782e-12))
	s = 0.0
	iter = 0

    while (error > tol && iter <= maxIter)
        error = 0.0
        for j = 1:t.Nx
            for i = 1:t.Ny
                old = t.phi[i,j]
                if (i == 1)
                    if (t.Nu == 0)
                        new_ = VG
                    else
                        if (j == 1)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j+1])
                        elseif (j == t.Nx)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j-1])
                        elseif (j > t.Nu && j <= t.Ng + t.Nu)
                            new_ = VG
                        else
                            new_ = 0.25*(2.0*t.phi[i+1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                        end
                    end
                elseif (j == 1)
                    if (i >= t.Nins + 1)
                        new_ = 0.0
                    else
                        new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + 2.0*t.phi[i,j+1])
                    end
                elseif (j == t.Nx)
                    if (i >= t.Nins + 1)
                        new_ = VD
                    else
                        new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + 2.0*t.phi[i,j-1])
                    end
                elseif (i == t.Ny)
                    new_ = 0.25*(2.0*t.phi[i-1,j] + t.phi[i,j-1] + t.phi[i,j+1])
                elseif (i == t.Nins + 1)
                    s = t.m.NDOP - t.n[j]
                    new_ = 0.25 * (k*s + t.phi[i,j+1] + t.phi[i,j-1] + t.phi[i-1,j] + t.phi[i+1,j])
                else
                    new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                end
                error += (old - new_)*(old - new_)
				t.phi[i,j] = sor*new_ + (1.0 - sor)*old
            end
        end
        error = sqrt(error / N)
        #println(error)
		iter += 1
    end
end

function solvePoisson!(t::DoubleGateTransistor, VG::Float64, VD::Float64, sor::Float64, tol::Float64, maxIter::Int)
    new_ = 0.0
	error = tol + 1.
	N = t.Nx * t.Ny
	k = (1.60217663e-19*t.dx / (t.m.eps * 8.85418782e-12))
	s = 0.0
	iter = 0

    while (error > tol && iter <= maxIter)
        error = 0.0
        for j = 1:t.Nx
            for i = 1:t.Ny
                old = t.phi[i,j]
                if (i == 1)
                    if (t.Nu == 0)
                        new_ = VG
                    else
                        if (j == 1)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j+1])
                        elseif (j == t.Nx)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j-1])
                        elseif (j > t.Nu && j <= t.Ng + t.Nu)
                            new_ = VG
                        else
                            new_ = 0.25*(2.0*t.phi[i+1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                        end
                    end
                elseif (i == t.Ny)
                    if (t.Nu == 0)
                        new_ = VG
                    else
                        if (j == 1)
                            new_ = 0.5*(t.phi[i-1,j] + t.phi[i,j+1])
                        elseif (j == t.Nx)
                            new_ = 0.5*(t.phi[i-1,j] + t.phi[i,j-1])
                        elseif (j > t.Nu && j <= t.Ng + t.Nu)
                            new_ = VG
                        else
                            new_ = 0.25*(2.0*t.phi[i-1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                        end
                    end
                elseif (j == 1)
                    if (i >= t.Nins + 1 && i <= t.Nins + t.Nsd + 1 )
                        new_ = 0.0
                    else
                        new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + 2.0*t.phi[i,j+1])
                    end
                elseif (j == t.Nx)
                    if (i >= t.Nins + 1 && i <= t.Nins + t.Nsd + 1 )
                        new_ = VD
                    else
                        new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + 2.0*t.phi[i,j-1])
                    end
                elseif (i == t.Nins + 1)
                    s = t.m.NDOP - t.n[j]
                    new_ = 0.25 * (k*s + t.phi[i,j+1] + t.phi[i,j-1] + t.phi[i-1,j] + t.phi[i+1,j])
                else
                    new_ = 0.25*(t.phi[i-1,j] + t.phi[i+1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                end
                error += (old - new_)*(old - new_)
				t.phi[i,j] = sor*new_ + (1.0 - sor)*old
            end
        end
        error = sqrt(error / N)
		iter += 1
    end
end

function solvePoisson!(t::DoubleChannelTransistor, VG::Float64, VD::Float64, sor::Float64, tol::Float64, maxIter::Int)
	new_ = 0.0
	error = tol + 1.
	N = t.Nx * t.Ny
	k = (1.60217663e-19*t.dx / (t.m.eps * 8.85418782e-12))
	s = 0.0
	iter = 0
	while (error > tol && iter < maxIter )
		error = 0.
		for j = 1:t.Nx
			for i = 1:t.Ny
				old = t.phi[i,j]
				if (i == 1)
					if (t.Nu == 0)
                        new_ = VG
                    else
                        if (j == 1)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j+1])
                        elseif (j == t.Nx)
                            new_ = 0.5*(t.phi[i+1,j] + t.phi[i,j-1])
                        elseif (j > t.Nu && j <= t.Ng + t.Nu)
                            new_ = VG
                        else
                            new_ = 0.25*(2.0*t.phi[i+1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                        end
                    end
				elseif (i == t.Ny)
					if (t.Nu == 0)
                        new_ = VG
                    else
                        if (j == 1)
                            new_ = 0.5*(t.phi[i-1,j] + t.phi[i,j+1])
                        elseif (j == t.Nx)
                            new_ = 0.5*(t.phi[i-1,j] + t.phi[i,j-1])
                        elseif (j > t.Nu && j <= t.Ng + t.Nu)
                            new_ = VG
                        else
                            new_ = 0.25*(2.0*t.phi[i-1,j] + t.phi[i,j+1] + t.phi[i,j-1])
                        end
                    end
				elseif (j == 1)
					if (i >= t.Nins + 1 && i <= t.Nins + t.Nsd + 2)
						new_ = 0.0
					else
						new_ = 0.25 * (t.phi[i+1,j] + t.phi[i-1,j] + 2.0*t.phi[i,j+1])
					end
				elseif (j == t.Nx)
					if (i >= t.Nins + 1 && i <= t.Nins + t.Nsd + 2)
						new_ = VD
					else
						new_ = 0.25 * (t.phi[i+1,j] + t.phi[i-1,j] + 2.0*t.phi[i,j-1])
					end
				elseif (i == t.Nins + 1)
					s = t.m.NDOP - t.n[j, 1]
					new_ = 0.25 * (k*s + t.phi[i,j+1] + t.phi[i,j-1] + t.phi[i-1,j] + t.phi[i+1,j])
				elseif (i == t.Nins + t.Nsd + 2)
					s = t.m.NDOP - t.n[j, 2] 
					new_ = 0.25 * (k*s + t.phi[i,j+1] + t.phi[i,j-1] + t.phi[i-1,j] + t.phi[i+1,j])
				else
					new_ = 0.25 * (t.phi[i,j+1] + t.phi[i,j-1] + t.phi[i-1,j] + t.phi[i+1,j])
				end
				error += (old - new_)*(old - new_)
				t.phi[i,j] = sor*new_ + (1.0 - sor)*old
			end
		end
		error = sqrt(error / N)
		iter += 1
		#println(error)
	end
	if (iter >= maxIter)
		@error "Cound not reach tol in iterations"
	end
end

function updateCarrierConcentration!(t::SingleChannelTransistor, VD::Float64, VG::Float64)

    if (t.m.ballistic)
        tobidx = argumentMin(t.phi_old, t.Nins + 1)
        phitob = t.phi_old[t.Nins + 1, tobidx]
        k2 = -t.m.D / t.m.VT
        k3 = -VD / t.m.VT
        k4 = phitob / t.m.VT
        
        for i=1:tobidx
            k1 = (t.phi_old[t.Nins + 1, i]) / t.m.VT
            t.n[i] = t.m.NC*(log(1+exp(k1+k2)) - 0.5*log(1+exp(k2+k4) - 0.5*log(1+exp(k2+k3+k4))))
        end

        for i=tobidx:t.Nx
            k1 = (t.phi_old[t.Nins + 1, i]) / t.m.VT
            t.n[i] = t.m.NC*(log(1+exp(k1+k2+k3)) - 0.5*log(1+exp(k2+k4+k3) - 0.5*log(1+exp(k2+k4))))
        end
    else 
        n0 =  t.m.NC*log(1.0 + exp((- t.m.D) / t.m.VT))
        t.ddlhs[1,1] = 1.0
        t.ddlhs[1,2] = 0.0
        t.ddlhs[t.Nx,t.Nx] = 1.0
        t.ddlhs[t.Nx, t.Nx-1] = 0.0
        t.ddrhs[1] = n0
        t.ddrhs[t.Nx] = n0
    
        for i=2:t.Nx-1
            k1 = (t.phi_old[t.Nins + 1, i+1] - t.phi_old[t.Nins + 1, i]) / (t.m.VT)
            k2 = (t.phi_old[t.Nins + 1, i] - t.phi_old[t.Nins + 1, i-1]) / (t.m.VT)
            t.ddlhs[i,i-1] = Ber(k2)*exp(k2)
            t.ddlhs[i,i] = -Ber(k2) - Ber(k1)*exp(k1)
            t.ddlhs[i,i+1] = Ber(k1)
            t.ddrhs[i] = 0.0
        end
        solveTridiagonal!(t.ddlhs, t.ddrhs, t.n)
    end

    return nothing
end



function calculateCurrent(t::SingleChannelTransistor)
    Javg = 0.0
    for i=1:t.Nx-1
        k = (t.phi_old[t.Nins + 1, i+1] - t.phi_old[t.Nins + 1, i]) / t.m.VT
        Javg += Ber(k)*(t.n[i+1] - t.n[i]*exp(k))
    end

    return (Javg / (t.Nx - 1))*(t.m.VT*1.6e-19*t.m.mu/t.dx)
end

function calculateCurrent(t::DoubleChannelTransistor)
    J1avg = 0.0
    J2avg = 0.0
    for i=1:t.Nx-1
        k = (t.phi_old[t.Nins + 1, i+1] - t.phi_old[t.Nins + 1, i]) / t.m.VT
        J1avg += Ber(k)*(t.n[i+1,1] - t.n[i,1]*exp(k))
        k = (t.phi_old[t.Nins + t.Nsd + 2, i+1] - t.phi_old[t.Nins + t.Nsd + 2, i]) / t.m.VT
        J2avg += Ber(k)*(t.n[i+1,2] - t.n[i,2]*exp(k))
    end

    return ((J1avg + J2avg) / (t.Nx - 1))*(t.m.VT*1.6e-19*t.m.mu/t.dx)
end

function updateCarrierConcentration!(t::DoubleChannelTransistor, VD::Float64, VG::Float64)
    n0 =  t.m.NC*log(1.0 + exp((- t.m.D) / t.m.VT))
    t.ddlhs[1,1] = 1.0
    t.ddlhs[1,2] = 0.0
    t.ddlhs[t.Nx,t.Nx] = 1.0
    t.ddlhs[t.Nx, t.Nx-1] = 0.0
    t.ddrhs[1] = n0
    t.ddrhs[t.Nx] = n0

    for i=2:t.Nx-1
        k1 = (t.phi_old[t.Nins + 1, i+1] - t.phi_old[t.Nins + 1, i]) / (t.m.VT)
        k2 = (t.phi_old[t.Nins + 1, i] - t.phi_old[t.Nins + 1, i-1]) / (t.m.VT)
        t.ddlhs[i,i-1] = Ber(k2)*exp(k2)
        t.ddlhs[i,i] = -Ber(k2) - Ber(k1)*exp(k1)
        t.ddlhs[i,i+1] = Ber(k1)
        t.ddrhs[i] = 0.0
    end

    solveTridiagonal!(t.ddlhs, t.ddrhs, view(t.n, :, 1))

    for i=2:t.Nx-1
        k1 = (t.phi_old[t.Nins + t.Nsd + 2, i+1] - t.phi_old[t.Nins + t.Nsd + 2, i]) / (t.m.VT)
        k2 = (t.phi_old[t.Nins + t.Nsd + 2, i] - t.phi_old[t.Nins + t.Nsd + 2, i-1]) / (t.m.VT)
        t.ddlhs[i,i-1] = Ber(k2)*exp(k2)
        t.ddlhs[i,i] = -Ber(k2) - Ber(k1)*exp(k1)
        t.ddlhs[i,i+1] = Ber(k1)
        t.ddrhs[i] = 0.0
    end

    solveTridiagonal!(t.ddlhs, t.ddrhs, view(t.n, :, 2))
end

function writeParameters(t::Transistor, filePrefix::String)
    fileName = filePrefix*"_parameters.json"
    open(fileName, "w") do f
        write(f, "{\n")
        write(f, "\"Device\" : {\n")
        write(f, "\"Lch\" : $(t.Lch),\n")
        write(f, "\"tins\" : $(t.tins),\n")
        write(f, "\"tsd\" : $(t.tsd),\n" )
        write(f, "\"dx\" : $(t.dx)\n")
        write(f, "},\n")
        write(f, "\"Material\" : {\n")
        write(f, "\"mn\" : $(t.m.mn),\n")
        write(f, "\"T\" : $(t.m.T),\n")
        write(f, "\"D\" : $(t.m.D),\n")
        write(f, "\"mu\" : $(t.m.mu),\n")
        write(f, "\"gn\" : $(t.m.gn)\n")
        write(f, "}\n")
        write(f, "}")
    end
end

function writeConductionBandProfile(t::Transistor, filePrefix::String)
    fileName = filePrefix*"_EC.csv"

    open(fileName, "w") do f
        write(f, "#x [nm], EC [eV]\n")
        for i=1:t.Nx
            if (i < t.Nx)
                write(f, "$((i-1)*t.dx*1e9), $(t.m.D - t.phi[t.Nins + 1, i])\n")
            else
                write(f, "$((i-1)*t.dx*1e9), $(t.m.D - t.phi[t.Nins + 1, i])")
            end
        end
    end
end

function outputSweep!(t::Transistor, VG, VD, tol::Float64, alpha::Float64, filePrefix::String)
    fileName = filePrefix*"_IDVD.csv"
    NVG = size(VG)[1]
    NVD = size(VD)[1]
    J = zeros(Float64, NVD, NVG)

    l = 1
    for vg in VG
        k = 1
        for vd in VD
            applyBias!(t, vg, vd, tol, alpha)
            j = calculateCurrent(t)
            println(j)
            J[k,l] = -1.0*j
            k = k + 1   
        end
        l += 1
    end

    open(fileName, "w") do f
        for j = 1:NVG
            if (j < NVG)
                write(f, "VG($(j)), VD($(j)), J($(j)),")
            else
                write(f, "VG($(j)), VD($(j)), J($(j))")
            end
    
        end
        write(f, "\n")
        for i = 1:NVD
            for j = 1:NVG
                if (j < NVG)
                    write(f, "$(VG[j]), $(VD[i]), $(J[i,j]),")
                else
                    write(f, "$(VG[j]), $(VD[i]), $(J[i,j])")
                end
            end
            write(f, "\n")
        end
    end

    return J
end

function inputSweep!(t::Transistor, VG, VD, tol::Float64, alpha::Float64, filePrefix::String)
    fileName = filePrefix*"_IDVG.csv"
    NVG = size(VG)[1]
    NVD = size(VD)[1]
    J = zeros(Float64, NVG, NVD)

    l = 1
    for vd in VD
        k = 1
        for vg in VG
            applyBias!(t, vg, vd, tol, alpha)
            j = calculateCurrent(t)
            println(j)
            J[k,l] = -1.0*j
            k = k + 1   
            println("-----------------")
        end
        l += 1
    end

    open("hello.txt", "w") do f 
        write(f, "Hello World\n")
    end

    open(fileName, "w") do f
        for j = 1:NVD
            if (j < NVD)
                write(f, "VG($(j)), VD($(j)), J($(j)),")
            else
                write(f, "VG($(j)), VD($(j)), J($(j))")
            end
        end
        write(f, "\n")
        for i = 1:NVG
            for j = 1:NVD
                if (j < NVD)
                    write(f, "$(VG[i]), $(VD[j]), $(J[i,j]),")
                else
                    write(f, "$(VG[i]), $(VD[j]), $(J[i,j])")
                end
            end
            write(f, "\n")
        end
    end

    return J
end

function applyBias!(t::Transistor, VG::Float64, VD::Float64, tol::Float64, alpha::Float64)
	error = tol + 1.0
	N = t.Nx*t.Ny
	while (error > tol)
		error = 0.0
		updateCarrierConcentration!(t, VD, VG)
		solvePoisson!(t, VG, VD, 1.5, 1e-12, 50000)
		for j = 1:t.Nx
			for i = 1:t.Ny
				error += (t.phi[i,j] - t.phi_old[i,j])*(t.phi[i,j] - t.phi_old[i,j])
				t.phi_old[i,j] = alpha*t.phi[i,j] + (1.0 - alpha)*t.phi_old[i,j]
			end
		end
		error = sqrt(error / N)	
		println(error)
	end
end

VG = LinRange(-0.75,1.0,72)
VD = [0.05,0.5]
m = Material(0.5, 0.5, 0.05, 300.0, 1.8, 0.01, 5.0, 4, 6, false )

t1 = DoubleGateTransistor(10e-9, 2e-9, 10e-9, 0e-9, m, 0.25e-9)
inputSweep!(t1, VG, VD, 5e-3, 5e-2, "DG10nm_10nm")

t2 = DoubleGateTransistor(10e-9, 2e-9, 8e-9, 0e-9, m, 0.25e-9)
inputSweep!(t2, VG, VD, 5e-3, 5e-2, "DG10nm_8nm")

t3 = DoubleGateTransistor(10e-9, 2e-9, 6e-9, 0e-9, m, 0.25e-9)
inputSweep!(t3, VG, VD, 5e-3, 5e-2, "DG10nm_6nm")

t4 = DoubleGateTransistor(10e-9, 2e-9, 4e-9, 0e-9, m, 0.25e-9)
inputSweep!(t4, VG, VD, 5e-3, 5e-2, "DG10nm_4nm")

t5 = DoubleGateTransistor(10e-9, 2e-9, 2e-9, 0e-9, m, 0.25e-9)
inputSweep!(t5, VG, VD, 5e-3, 5e-2, "DG10nm_2nm")

t6 = DoubleGateTransistor(10e-9, 2e-9, 1e-9, 0e-9, m, 0.25e-9)
inputSweep!(t6, VG, VD, 5e-3, 5e-2, "DG10nm_1nm")
