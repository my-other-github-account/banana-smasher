def run_pair(paths, root, layer, public):
 assert len(paths)==2 and len(set(map(str,paths)))==2
 return public(paths,root,layer)
