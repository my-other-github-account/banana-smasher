#!/usr/bin/env python3
"""Audit exact source overlap against a pinned EXL3 source tree.

Inputs are explicit paths; this tool never copies upstream files into the candidate tree.
It reports exact files, normalized lines, exact 20/40-token shingles, and reachable
Git-history blobs for subsequent manual classification.
"""
import argparse, ast, hashlib, json, os, re, subprocess, tokenize, io
from pathlib import Path
from collections import defaultdict

EXTS={'.py','.pyi','.cu','.cuh','.cpp','.cc','.cxx','.c','.h','.hpp','.rs','.js','.ts','.sh','.bash'}
PRODUCT_PREFIXES=('banana-smasher/src/','banana-smasher-plugin/src/','runtime/','docker/scripts/')
SKIP_DIRS={'.git','__pycache__','.pytest_cache','node_modules','build','dist'}
GENERIC_LINES={
 'from__future__importannotations','importos','importjson','importtorch','importnumpyasnp',
 'returnnone','returntrue','returnfalse','if__name__=="__main__":','if__name__==\'__main__\':'
}

def sha256(b): return hashlib.sha256(b).hexdigest()
def digest_shingle(tokens, i, n):
    return hashlib.blake2b('\x1f'.join(tokens[i:i+n]).encode(), digest_size=16).digest()

def strip_c_comments(text):
    text=re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text=re.sub(r'//[^\n]*', ' ', text)
    return text

def generic_tokens(text):
    text=strip_c_comments(text)
    # Preserve operators, identifiers, and numeric literals; string prose becomes STR.
    text=re.sub(r'"(?:\\.|[^"\\])*"', ' STR ', text, flags=re.S)
    text=re.sub(r"'(?:\\.|[^'\\])*'", ' CHR ', text, flags=re.S)
    return re.findall(r'[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fFlLuU]*)|<<=|>>=|==|!=|<=|>=|->|::|\+\+|--|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\.\.\.|[^\s]', text)

def python_tokens(text):
    out=[]
    try:
        stream=tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in stream:
            if tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.OP):
                out.append(tok.string)
            elif tok.type==tokenize.STRING:
                try: value=ast.literal_eval(tok.string)
                except Exception: continue
                if isinstance(value,str) and ('\n' in value or '__global__' in value or '#include' in value or 'template<' in value):
                    out.extend(generic_tokens(value))
    except (tokenize.TokenError,IndentationError,SyntaxError):
        return generic_tokens(text)
    return out

def tokens_for(path,text):
    return python_tokens(text) if Path(path).suffix in {'.py','.pyi'} else generic_tokens(text)

def normalized_lines(text):
    rows=[]
    in_block=False
    for lineno, raw in enumerate(text.splitlines(),1):
        s=raw.strip()
        if not s: continue
        # Keep comments in the line audit because copied comments are provenance evidence.
        norm=''.join(s.split())
        norm=norm.strip('`')
        if len(norm)<24: continue
        words=re.findall(r'[A-Za-z_]\w*|\d+|[^\w\s]', norm)
        if len(words)<5 or norm.lower() in GENERIC_LINES: continue
        rows.append((norm,lineno,s[:500]))
    return rows

def iter_files(root):
    root=Path(root)
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXTS: continue
        if any(part in SKIP_DIRS for part in p.parts): continue
        rel=p.relative_to(root).as_posix()
        try: b=p.read_bytes(); text=b.decode('utf-8')
        except Exception: continue
        yield rel,b,text

def scenario_scope(rel):
    return 'product' if rel.startswith(PRODUCT_PREFIXES) else 'nonshipping_test_tool_doc'

def build_upstream(root):
    files={}; line_map=defaultdict(list); exact_map=defaultdict(list); shingle={20:{},40:{}}
    token_count=0; line_count=0
    for rel,b,text in iter_files(root):
        toks=tokens_for(rel,text); token_count+=len(toks)
        lines=normalized_lines(text); line_count+=len(lines)
        files[rel]={'sha256':sha256(b),'tokens':len(toks),'lines':len(lines)}
        exact_map[sha256(b)].append(rel)
        for norm,ln,raw in lines: line_map[norm].append((rel,ln,raw))
        for n in (20,40):
            if len(toks)<n: continue
            m=shingle[n]
            for i in range(len(toks)-n+1):
                d=digest_shingle(toks,i,n)
                if d not in m: m[d]=(rel,i,toks[i:i+n])
    return {'files':files,'line_map':line_map,'exact_map':exact_map,'shingle':shingle,'tokens':token_count,'meaningful_lines':line_count}

def compress_matches(raw,n):
    raw.sort(key=lambda x:(x[0],x[1],x[2]))
    groups=[]
    for cand_i,up_file,up_i,seq in raw:
        if groups and groups[-1]['upstream_file']==up_file and cand_i==groups[-1]['candidate_end'] and up_i==groups[-1]['upstream_end']:
            groups[-1]['candidate_end']+=1; groups[-1]['upstream_end']+=1; groups[-1]['matched_tokens']+=1
        else:
            groups.append({'candidate_start':cand_i,'candidate_end':cand_i+1,'upstream_file':up_file,'upstream_start':up_i,'upstream_end':up_i+1,'matched_tokens':n,'first_shingle':seq})
    return groups

def compare_text(rel,b,text,up):
    toks=tokens_for(rel,text); result={'path':rel,'scope':scenario_scope(rel),'sha256':sha256(b),'tokens':len(toks)}
    exact=up['exact_map'].get(result['sha256'],[])
    if exact: result['exact_file_matches']=exact
    line_matches=[]
    for norm,ln,raw in normalized_lines(text):
        for urel,uln,uraw in up['line_map'].get(norm,[]):
            line_matches.append({'candidate_line':ln,'candidate_text':raw,'upstream_file':urel,'upstream_line':uln,'upstream_text':uraw,'normalized':norm})
    if line_matches: result['line_matches']=line_matches[:500]
    for n in (20,40):
        raw=[]; m=up['shingle'][n]
        if len(toks)>=n:
            for i in range(len(toks)-n+1):
                d=digest_shingle(toks,i,n); hit=m.get(d)
                if hit and toks[i:i+n]==hit[2]: raw.append((i,hit[0],hit[1],hit[2]))
        groups=compress_matches(raw,n)
        if groups: result[f'shingle_{n}_groups']=groups[:200]
    mentions=[]
    for ln,s in enumerate(text.splitlines(),1):
        if re.search(r'(?i)exllama|exl3|791c830',s): mentions.append({'line':ln,'text':s[:500]})
    if mentions: result['exl_mentions']=mentions
    return result

def compare_scenario(root,up):
    rows=[]; totals={'files':0,'tokens':0,'product_files':0,'nonshipping_files':0,'exact_files':0,'line_matches':0,'shingle20_groups':0,'shingle40_groups':0,'exl_mentions':0}
    for rel,b,text in iter_files(root):
        r=compare_text(rel,b,text,up); rows.append(r); totals['files']+=1; totals['tokens']+=r['tokens'];totals['product_files' if r['scope']=='product' else 'nonshipping_files']+=1
        totals['exact_files']+=int('exact_file_matches' in r); totals['line_matches']+=len(r.get('line_matches',[])); totals['shingle20_groups']+=len(r.get('shingle_20_groups',[])); totals['shingle40_groups']+=len(r.get('shingle_40_groups',[])); totals['exl_mentions']+=len(r.get('exl_mentions',[]))
    interesting=[r for r in rows if any(k in r for k in ('exact_file_matches','line_matches','shingle_20_groups','shingle_40_groups','exl_mentions'))]
    return {'totals':totals,'interesting':interesting,'file_hashes':{r['path']:r['sha256'] for r in rows}}

def history_blobs(repo):
    cp=subprocess.run(['git','rev-list','--objects','--all'],cwd=repo,text=True,capture_output=True,check=True)
    items=[]; seen=set()
    for line in cp.stdout.splitlines():
        parts=line.split(' ',1)
        if len(parts)!=2: continue
        sha,path=parts
        if Path(path).suffix.lower() not in EXTS: continue
        if sha in seen: continue
        seen.add(sha);items.append((sha,path))
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    out=[]
    assert proc.stdin and proc.stdout
    for sha,path in items:
        proc.stdin.write((sha+'\n').encode())
    proc.stdin.close()
    for sha,path in items:
        header=proc.stdout.readline().decode().strip().split()
        if len(header)<3 or header[1]!='blob':
            continue
        size=int(header[2]);b=proc.stdout.read(size);proc.stdout.read(1)
        try:text=b.decode('utf-8')
        except Exception:continue
        yield sha,path,b,text
    proc.wait()

def compare_history(repo,up):
    interesting=[]; totals={'unique_source_blobs':0,'tokens':0,'exact_files':0,'line_matches':0,'shingle20_groups':0,'shingle40_groups':0,'exl_mentions':0}
    for sha,path,b,text in history_blobs(repo):
        totals['unique_source_blobs']+=1
        r=compare_text(path,b,text,up);r['blob_sha']=sha;totals['tokens']+=r['tokens'];totals['exact_files']+=int('exact_file_matches'in r);totals['line_matches']+=len(r.get('line_matches',[]));totals['shingle20_groups']+=len(r.get('shingle_20_groups',[]));totals['shingle40_groups']+=len(r.get('shingle_40_groups',[]));totals['exl_mentions']+=len(r.get('exl_mentions',[]))
        if any(k in r for k in ('exact_file_matches','line_matches','shingle_20_groups','shingle_40_groups','exl_mentions')):interesting.append(r)
    return {'totals':totals,'interesting':interesting}

def main():
    a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--canonical',required=True);a.add_argument('--fanin',required=True);a.add_argument('--repo',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    up=build_upstream(args.upstream)
    result={'schema':'banana-exl-implementation-independence-audit-v1','upstream':{'root':args.upstream,'revision':'791c83073f7f90c44f765a0ceeab7a05fa15b96b','files':len(up['files']),'tokens':up['tokens'],'meaningful_lines':up['meaningful_lines']},'canonical':compare_scenario(args.canonical,up),'fanin':compare_scenario(args.fanin,up),'history':compare_history(args.repo,up),'method':{'extensions':sorted(EXTS),'exact_file_sha256':True,'normalized_meaningful_lines':True,'exact_token_shingles':[20,40],'comments_removed_for_token_scan':True,'embedded_python_multiline_source_strings_tokenized':True,'history_scope':'all source blobs reachable from all refs'}}
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':args.output,'upstream':result['upstream'],'canonical':result['canonical']['totals'],'fanin':result['fanin']['totals'],'history':result['history']['totals']},indent=2))
if __name__=='__main__':main()
