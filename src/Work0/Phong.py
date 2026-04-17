import taichi as ti
ti.init(arch=ti.gpu)
width = 800
height = 600
img = ti.Vector.field(3, dtype=float, shape=(width, height))
ambient_coeff = ti.field(float, shape=())
diffuse_coeff = ti.field(float, shape=())
specular_coeff = ti.field(float, shape=())
shininess_val = ti.field(float, shape=())

@ti.func
def unit(v):
    return v.normalized(1e-6)

@ti.func
def reflect_vec(inc, norm):
    return inc - 2 * inc.dot(norm) * norm

@ti.func
def sphere_hit(ray_origin, ray_dir, center, radius):
    dist = -1.0
    norm = ti.Vector([0.0, 0.0, 0.0])
    oc = ray_origin - center
    a = 1.0
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4 * a * c
    if disc > 0:
        t = (-b - ti.sqrt(disc)) / (2 * a)
        if t > 1e-6:
            dist = t
            pt = ray_origin + ray_dir * t
            norm = unit(pt - center)
    return dist, norm

@ti.func
def cone_hit(ray_origin, ray_dir, apex, base_y, radius):
    dist = -1.0
    norm = ti.Vector([0.0, 0.0, 0.0])
    height = apex.y - base_y
    ratio = (radius / height) ** 2
    local_ro = ray_origin - apex
    A = ray_dir.x**2 + ray_dir.z**2 - ratio * ray_dir.y**2
    B = 2 * (local_ro.x*ray_dir.x + local_ro.z*ray_dir.z - ratio*local_ro.y*ray_dir.y)
    C = local_ro.x**2 + local_ro.z**2 - ratio*local_ro.y**2
    if ti.abs(A) > 1e-6:
        disc = B*B - 4*A*C
        if disc > 0:
            t1 = (-B - ti.sqrt(disc)) / (2*A)
            t2 = (-B + ti.sqrt(disc)) / (2*A)
            t_near = t1 if t1 < t2 else t2
            t_far = t2 if t1 < t2 else t1
            y_check = local_ro.y + t_near * ray_dir.y
            if t_near > 1e-6 and -height <= y_check <= 0:
                dist = t_near
            else:
                y_check2 = local_ro.y + t_far * ray_dir.y
                if t_far > 1e-6 and -height <= y_check2 <= 0:
                    dist = t_far
            if dist > 1e-6:
                pt_local = local_ro + ray_dir * dist
                norm = unit(ti.Vector([pt_local.x, -ratio*pt_local.y, pt_local.z]))
    return dist, norm

@ti.kernel
def draw():
    for x, y in img:
        u = (x - width/2) / height * 2
        v = (y - height/2) / height * 2
        cam_pos = ti.Vector([0.0, 0.0, 5.0])
        ray_dir = unit(ti.Vector([u, v, -1.0]))
        closest = 1e10
        final_norm = ti.Vector([0.0,0.0,0.0])
        final_col = ti.Vector([0.0,0.0,0.0])
        
        t1, n1 = sphere_hit(cam_pos, ray_dir, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t1 < closest:
            closest = t1
            final_norm = n1
            final_col = ti.Vector([0.8, 0.1, 0.1])
            
        t2, n2 = cone_hit(cam_pos, ray_dir, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t2 < closest:
            closest = t2
            final_norm = n2
            final_col = ti.Vector([0.6, 0.2, 0.8])
            
        col = ti.Vector([0.04, 0.16, 0.18])
        if closest < 1e9:
            hit_pt = cam_pos + ray_dir * closest
            N = final_norm
            light_p = ti.Vector([2.0, 3.0, 4.0])
            light_c = ti.Vector([1.0, 1.0, 1.0])
            L = unit(light_p - hit_pt)
            V = unit(cam_pos - hit_pt)
            amb = ambient_coeff[None] * light_c * final_col
            diff_val = ti.max(0.0, N.dot(L))
            diff = diffuse_coeff[None] * diff_val * light_c * final_col
            R = unit(reflect_vec(-L, N))
            spec_val = ti.max(0.0, R.dot(V)) ** shininess_val[None]
            spec = specular_coeff[None] * spec_val * light_c
            col = amb + diff + spec
        img[x,y] = ti.math.clamp(col, 0.0, 1.0)

def run():
    win = ti.ui.Window("Phong Lighting", (width, height))
    canvas = win.get_canvas()
    ui = win.get_gui()
    
    ambient_coeff[None] = 0.2
    diffuse_coeff[None] = 0.7
    specular_coeff[None] = 0.5
    shininess_val[None] = 32.0
    
    while win.running:
        draw()
        canvas.set_image(img)
        with ui.sub_window("Controls", 0.7, 0.05, 0.28, 0.23):
            ambient_coeff[None] = ui.slider_float("Ka", ambient_coeff[None], 0.0, 1.0)
            diffuse_coeff[None] = ui.slider_float("Kd", diffuse_coeff[None], 0.0, 1.0)
            specular_coeff[None] = ui.slider_float("Ks", specular_coeff[None], 0.0, 1.0)
            shininess_val[None] = ui.slider_float("Shininess", shininess_val[None], 1.0, 128.0)
        win.show()
if __name__ == "__main__":
    run()