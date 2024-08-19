using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
#using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt, @vv etc.
using LinearAlgebra

function main()
    use_cuda = false;

    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_directions = 4; num_images =  3*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    use_lattice = false #seq=sequential
    seq_lattice=true #In the follwoing a continous shift of the lattic is implemented. 
  

    num_photons = 1000.00
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
    cont_cond = cond(peak_strengths .* exp.(1im .* peak_phases))
    #seq_cond = cond()
   
    #The following lattice shift is continous  
    #To find the perfect shift size a loop is implementen. To finde the best solution of the shift size the conditional analysis is used. A good value for the solution is a value between 1 and 2. (Anyway as small as possible)
    N = 100;
    res = zeros(N,N) #N*N Matrix of zeros
    kmax = 10.0; 
 
    for x = 0:N-1 #looks at values in x direction
        for y = 0:N-1 #looks at values in y direction
            @show x,y
            lattice_shift = [kmax*x/N, kmax*y/N, 0.0]
            k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1); use_lattice=use_lattice, lattice_shift=lattice_shift)

            num_photons = 1000.00
            spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
            res[x+1,y+1] = cond(peak_strengths .* exp.(1im .* peak_phases))
            @show findmin(res)
        end
    end
   
    

    for x = 0:N-1 #looks at values in x direction, optimierung der shift größe
        for y = 0:N-1 #looks at values in y direction
            k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1);seq_lattice=seq_lattice)
            num_photons = 1000.00
            spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
            res[x+1,y+1] = cond(peak_strengths .* exp.(1im .* peak_phases))
        end
    end
   
    @show mymin,myminpos=findmin(res) #find min value of conditional 
    @vv res   #shows min value of conditional
    lattice_shift = [kmax*(myminpos[1]-1)/N, kmax*(myminpos[2]-1)/N, 0.0] #set minimal conditional in matrix
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1); seq_lattice=seq_lattice, use_lattice=use_lattice, lattice_shift=lattice_shift)

    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
            

    obj = Float32.(testimage("resolution_test_512")); #load test image
    obj[(size(obj).÷2 .+1)...] = 2.0  #shows SIM Pattern with 9 pictures 

    illu = SIMPattern.(Ref(obj), Ref(spf), 1:12, 1) #SIM pattern with 1:9 images, the ref() signalises value stays as before, 
    @vt illu #shows SIM Pattern with 9 pictures 

    if (false)
        obj .= 0.0
        # obj[257,257] = 1.0
        obj[250,250] = 1.0
    end
    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    downsample_factor = 2

    @time sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
    
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    #################################

    # @vv sim_data
    rp = ReconParams() # just use defaults
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 1e-4
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    #rp.use_measure=!use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.use_hgoal = true
    rp.hgoal_exp = 0.5
    prep = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate

    @time recon = recon_sim(sim_data, prep, sp);
    wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt obj wf recon 

    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  # 480 µs (one zero order, 256x256)
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 2.2 ms (one zero order, 256x256)
    end

end
#@vv otf
