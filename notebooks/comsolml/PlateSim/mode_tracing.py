from typing import Sequence
import numpy as np
from scipy import interpolate
from scipy import optimize
from .plate import Plate
from .characteristic_equations import Ab, Sb

pi = np.pi
nax = np.newaxis

def cutoff_f_A(solid, d, n):
    cS = solid.cS
    cP = solid.cP
    
    nP = np.arange(0,n+1,dtype = np.float)
    nS = 1/2 * (2*np.arange(1,n+1,dtype = np.float) - 1)
    
    fcP = nP*cP/d
    fcS = nS*cS/d
    
    res = np.append(fcP,fcS)
    return np.sort(res)[0:n+1]

def cutoff_f_S(solid, d, n):
    cS = solid.cS
    cP = solid.cP
    
    nS =  np.arange(0,n+1, dtype = np.float)
    nP = 1/2 * (2*np.arange(1,n+1,dtype = np.float) - 1)
    
    fcP = nP*cP/d
    fcS = nS*cS/d
    
    res = np.append(fcP,fcS)
    return np.sort(res)[0:n+1]


def zero_list(d, cP, cS, w_min, w_max, n, kx, func):
    """Find zero w's of func for given kx, using n points"""
    zero_list = []
    funcw = lambda w: func(w, d, kx, cP, cS)
    ws = np.linspace(w_min, w_max, n)
    for i in range(len(ws)-1):
        if funcw(ws[i+1])*funcw(ws[i]) < 0:
            root = optimize.root_scalar(funcw, method='brentq', bracket=[ws[i], ws[i+1]]).root
            if abs(funcw(root)) < 1e-3:
                zero_list.append(root)
    return zero_list

def trace_direction(direction, kxs, ws, func, w_min, w_max, b_min, b_max, d, cS, cP, db = 5): ###
    """ Trace a mode from 2 start values of kx and w in either up(+) or down(-) direction """
    if direction == 'up':
        sign = +1
    if direction == 'down':
        sign = -1
        
    reached_end = False
    while not reached_end:
        dw = (ws[-1] - ws[-2])/(kxs[-1] - kxs[-2])*db # linear extrapolation of w
        w0 = ws[-1] + sign*dw
        b0 = kxs[-1] + sign*db
        
        beyond_limits = (w0 <= w_min) or (w0 >= w_max) or (b0 <= b_min) or (b0 >= b_max)
        if beyond_limits: break
        
        # else look for new root
        funcw = lambda w: func(w, d, b0, cP, cS)
        try: new_w = optimize.newton(funcw, x0=w0)
        except: break
        
        jump_too_big = (abs(new_w - w0) > abs(5*dw))
        if jump_too_big: break
        # else we append to list (by reference, so need to return)   
        ws.append(new_w)
        kxs.append(b0)

def trace_mode(kxs, ws, func, w_min, w_max, b_min, b_max, d, cS, cP, db = 5):
    """ Trace a given mode """
    # Trace mode up from kxs0 until break condition. Stepwise db.
    trace_direction('up', kxs, ws, func, w_min, w_max, b_min, b_max, d, cS, cP, db = 5)
    ws.reverse()
    kxs.reverse()
    
    # down
    trace_direction('down', kxs, ws, func, w_min, w_max, b_min, b_max, d, cS, cP, db = 5)
    ws.reverse()
    kxs.reverse()
    return np.array(kxs), np.array(ws)



        
class LambMode_piece:
    def __init__(self, bs,ws, b0, w0):   
        self.kx_min = min(bs)
        self.kx_max = max(bs)
        self.w_min = min(ws)
        self.w_max = max(ws)
        self.w0 = w0
        self.b0 = b0
        self.ws = ws
        self.bs = bs

    def extend(self, b_new, w_new):
        w_new = w_new[ (b_new < self.kx_min) | (b_new > self.kx_max) ]
        b_new = b_new[ (b_new < self.kx_min) | (b_new > self.kx_max) ]
        
        b_new = np.append(b_new, self.bs)
        w_new = np.append(w_new, self.ws)
        inds_sorted = b_new.argsort()
        
        self.bs = b_new[inds_sorted]
        self.ws = w_new[inds_sorted]
        
        self.kx_min = min(self.bs)
        self.kx_max = max(self.bs)
        self.w_min = min(self.ws)
        self.w_max = max(self.ws)

def trace_from_b(b0, w_min, w_max,b_min, b_max, d, cS, cP, func):
    n = 200
    modes = {}
    zb0 = zero_list(d, cP, cS, w_min, w_max, n, b0, func)
    zb1 = zero_list(d, cP, cS, w_min, w_max, n, b0+1e-5, func)
    assert len(zb0) == len(zb1), "Different number of modes found at initial kxs. Try change w_min / w_max."
    for i in range(len(zb0)):
        bs,ws = trace_mode([b0,b0+1e-5], [zb0[i],zb1[i]], func, w_min, w_max, b_min, b_max, d, cS, cP, db = 5)
        modes['M'+str(i)] = LambMode_piece(bs, ws, b0, zb0[i])
    return modes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def best_match_distance(mode_piece, dict2): # Legg inn sjekk for at de deriverte matcher! ( kryssende moder = :c )
    if dict2 == {}: return 'Empty', 0
    
    w0 = mode_piece.ws[-1]
    b0 = mode_piece.bs[-1]
    
    y1L = mode_piece.ws[-2]
    y2L = mode_piece.ws[-1]
    x1L = mode_piece.bs[-2]
    x2L = mode_piece.bs[-1]
    deriv_line = (y2L - y1L)/(x2L - x1L)

    max_diff_b = 13
    max_diff_w = max_diff_b * deriv_line * 1.5
    
    for key in dict2:
        i_b  = find_nearest(dict2[key].bs , b0)
        
        b1 = dict2[key].bs[i_b]
        w1 = dict2[key].ws[i_b]
        
        y1R = dict2[key].ws[i_b]
        y2R = mode_piece.ws[i_b+1]
        x1R = mode_piece.bs[i_b]
        x2R = mode_piece.bs[i_b+1]
        deriv_piece = (y2R - y1R)/(x2R - x1R)

        angle = abs(np.arctan((deriv_piece-deriv_line)/(1+deriv_piece*deriv_line + 1e-5))*180/pi)
        if (abs(w0-w1) < max_diff_w) & (abs(b0-b1) < max_diff_b) & (angle < 5):
            return key, i_b
 
    return 'No match, not empty', 0


#def best_match_overlap(mode_piece, dict2):
#    wend = mode_piece.ws[-1]
#    bend = mode_piece.bs[-1]
#    
#    if dict2 == {}: return 'Empty', 0
#    
#    w_max_diff = 100
#    b_max_diff = 7
#
#    for key in dict2:
#        w0  = dict2[key].w0
#        b0  = dict2[key].b0
#        if (abs(wend-w0) < w_max_diff) & (abs(bend-b0) < b_max_diff):
#            return key
#    return 'No match, not empty'

def best_match_intersection(mode_piece, dict2):
    if dict2 == {}: return 'Empty'
    
    y1L = mode_piece.ws[-2]
    y2L = mode_piece.ws[-1]
    x1L = mode_piece.bs[-2]
    x2L = mode_piece.bs[-1]
    
    aL = (y2L - y1L)/(x2L - x1L)
    bL = y1L - aL*x1L
    
    closest = 1e9
    closest_key = None
    
    for key in dict2:
        y1R = dict2[key].ws[0]
        y2R = dict2[key].ws[1]
        x1R = dict2[key].bs[0]
        x2R = dict2[key].bs[1]        

        aR = (y2R - y1R)/(x2R - x1R)
        bR = y1R - aR*x1R
        
        #x_intersection = (bR - bL)/(aL - aR)
        x_avg = (x1L + x1R)/2
        y_avg = (y1L + y1R)/2
        
        y_diffR = abs(aR*x_avg + bR - y_avg) #,  abs(aR*x_avg + bR - y_avg))
        y_diffL = abs(aL*x_avg + bL - y_avg)
        y_diff = y_diffR + y_diffL
        
        if (y_diff < closest) & (y_diff < 0.5*(x1R-x1L)*(abs(aL)+abs(aR))):
            closest_key = key
            closest = abs(y_diff)
    if closest_key == None: return 'No match, not empty' # should not be possible...
    return closest_key

def match_mode_pieces(list_mode_pieces_dict, d,cS):
    number_of_bs = len(list_mode_pieces_dict) # number of bs in b0s (len of list, list of dicts)
    lines = list_mode_pieces_dict[0].copy() # first part of modes, call each a line
    
    for i in range(1, number_of_bs): # for all remaining cuts in b0s
        unmatched_lines = []
        for key in lines:
            match_key, iw = best_match_distance( lines[key], list_mode_pieces_dict[i])
            if match_key == 'Empty': break # no more things to compare
            elif match_key == 'No match, not empty':# continue
                unmatched_lines.append(key)
            else:
                lines[key].extend(list_mode_pieces_dict[i][match_key].bs[iw:], list_mode_pieces_dict[i][match_key].ws[iw:])
                list_mode_pieces_dict[i].pop(match_key)
        
        for key in unmatched_lines:
            match_key = best_match_intersection( lines[key], list_mode_pieces_dict[i])
         
            if match_key == 'Empty': break # no more things to compare
            elif match_key == 'No match, not empty': continue
            else:
                lines[key].extend(list_mode_pieces_dict[i][match_key].bs, list_mode_pieces_dict[i][match_key].ws)
                list_mode_pieces_dict[i].pop(match_key)
                   
        if list_mode_pieces_dict[i] != {}:
            for remaining_key in list_mode_pieces_dict[i]:
                lines[remaining_key + str(i)] = list_mode_pieces_dict[i][remaining_key]
    return lines
        
    
#[10,50,100,200,400,600,800,1000,1200,1400,1600] 
def mode_dict(w_min: float, w_max: float, plate: Plate,
              b0s: Sequence[float] = [50,100,200,400,800,1000,1400,1600,2000,3000] ):
    modes = {}
    d = plate.d
    cP = plate.material.cP
    cS = plate.material.cS

    # Antisymmetric
    Mode_pieces = []
    for i, b0 in enumerate(b0s):
        if i == 0: 
            b_min = 0.
            b_max = b0s[1]
        elif (i+1) == len(b0s):
            b_min = b0s[i-1]
            b_max = 1.e9
        else:
            b_min = b0s[i-1]
            b_max = b0s[i+1]
        Mode_pieces.append(  trace_from_b(b0, w_min, w_max, b_min, b_max, d, cS, cP, Ab)  )
    
    modes_matched = match_mode_pieces(Mode_pieces, d,cS)
    
    temp_names = []
    temp_w = []
    for key in modes_matched:
        temp_names.append(key)
        temp_w.append( modes_matched[key].w_min )
        # consider filtering out based on range of w's and b's, so lone pieces are removed
    sorted_w, sorted_name = map(list, zip(*sorted(zip(temp_w, temp_names))))

    num_del = 0
    for i in range(len(sorted_w)):
        modes['A' + str(i - num_del)] = LambMode(modes_matched[ sorted_name[i] ].bs, modes_matched[ sorted_name[i] ].ws)
        if modes['A' + str(i- num_del)].spline_b_from_w == "delete mode":
            del modes['A' + str(i- num_del)] 
            num_del += 1   
    
    # Symmetric
    Mode_pieces = []
    for i, b0 in enumerate(b0s):
        if i == 0: 
            b_min = 0.
            b_max = b0s[1]
        elif (i+1) == len(b0s):
            b_min = b0s[i-1]
            b_max = 1.e9
        else:
            b_min = b0s[i-1]
            b_max = b0s[i+1]
        Mode_pieces.append(  trace_from_b(b0, w_min, w_max, b_min, b_max, d, cS, cP, Sb)  )
    
    modes_matched = match_mode_pieces(Mode_pieces, d,cS)
    temp_names = []
    temp_w = []
    for key in modes_matched:
        temp_names.append(key)
        temp_w.append( modes_matched[key].w_min )
    sorted_w, sorted_name = map(list, zip(*sorted(zip(temp_w, temp_names))))
    
    num_del = 0
    for i in range(len(sorted_w)):
        modes['S' + str(i- num_del)] = LambMode(modes_matched[ sorted_name[i] ].bs, modes_matched[ sorted_name[i] ].ws)        
        if modes['S' + str(i- num_del)].spline_b_from_w == "delete mode":
            del modes['S' + str(i- num_del)]
            num_del += 1
    return modes 



class LambMode:
    def __init__(self, bs,ws):   
        self.kx_min = min(bs)
        self.kx_max = max(bs)
        self.w_min = min(ws)
        self.w_max = max(ws)
        self.ws = ws
        self.bs = bs
        
        self.spline_w_from_b = interpolate.splrep(bs, ws)
        
        arg_wmin = np.argmin(ws)
        sort_ind = np.argsort(ws[arg_wmin:]) 
        wshort = (ws[arg_wmin:])[sort_ind]
        bshort = (bs[arg_wmin:])[sort_ind]
        
        if len(wshort) > 5:
            self.spline_b_from_w = interpolate.splrep(wshort, bshort)
        else:
            self.spline_b_from_w = "delete mode"
            
        #_, ind_wunique = np.unique(wshort, return_index=True)
        #bshort = bshort[ind_wunique]
        #wshort = wshort[ind_wunique]
        
         
        
    def w_from_kx(self, kx):
        #kx = kx[ (kx > kx_min) & (kx < kx_max) ]
        res = interpolate.splev(kx, self.spline_w_from_b, der = 0)
        if np.isscalar(kx) == True:
            return res
        res[res < 0] = 0
        return res
    def kx_from_w(self, w):
        #w = w[ (w > w_min) & (w < w_max) ]
        res = interpolate.splev(w, self.spline_b_from_w, der = 0)
        if np.isscalar(w) == True:
            return res
        res[res < 0] = 0
        return res
    def cgr_from_w(self, w):
        #w = w[(w > w_min) & (w < w_max) ]
        res = 1/interpolate.splev(w, self.spline_b_from_w, der = 1)
        if np.isscalar(w) == True:
            return res
        res[res < 0] = 10000
        return res
    def cph_from_w(self, w):
        #w = w[ (w > self.w_min) & (w < self.w_max) ]
        res = w / interpolate.splev(w, self.spline_b_from_w, der = 0)
        if np.isscalar(w) == True:
            return res
        res[res < 0] = 50000
        return res
