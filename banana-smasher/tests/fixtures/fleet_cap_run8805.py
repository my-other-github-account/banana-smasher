G=1024**3
def budget(cells):
 assert len(cells)==1 and cells[0].rsplit('_',1)[-1] in ('down','fused13')
 return 56*G,(10 if cells[0].endswith('fused13') else 6)*1024**2
def verify(s,available,free):
 peak,output=budget(s['cells']);assert s['estimated_peak_bytes']==peak and s['planned_write_bytes']>=output
 assert available>peak+8*G+s['planned_write_bytes'],'HOST_8G_HEADROOM_REFUSED'
 assert free>4*G+s['planned_write_bytes'],'ORIGINAL_STORAGE_RESERVE_REFUSED'
 return peak,output
def apply_cap(s,resource,cuda):
 peak,output=budget(s['cells']);assert s['estimated_peak_bytes']==peak
 resource.setrlimit(resource.RLIMIT_AS,(48*G,48*G));cuda.set_per_process_memory_fraction(4*G/cuda.get_device_properties(0).total_memory,0)
 return dict(cpu_address_limit=48*G,cuda_allocator_limit=4*G,overhead_bytes=4*G,estimated_peak_bytes=peak,numerical_change=False,scope='source-localized singleton on redistributed high-MemAvailable seat')
