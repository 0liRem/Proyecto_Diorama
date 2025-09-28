use minifb::{Window, WindowOptions, Key};
use std::f32::consts::PI;
use std::ops::{Add, Sub, Mul};
use std::time::{Instant, Duration};
use image::{io::Reader as ImageReader};
use std::path::Path;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

const WIDTH: usize = 640;
const HEIGHT: usize = 480; 

#[derive(Copy, Clone)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(&self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        } else {
            *self
        }
    }

    fn subtract(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn multiply(&self, scalar: f32) -> Vec3 {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    fn multiply_vec(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, scalar: f32) -> Vec3 {
        self.multiply(scalar)
    }
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }

    fn point_at_parameter(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    fov: f32,
    aspect_ratio: f32,
}

impl Camera {
    fn new(position: Vec3, lookat: Vec3, vup: Vec3, fov: f32, aspect_ratio: f32) -> Self {
        let theta = fov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        
        let w = (position - lookat).normalize();
        let u = vup.cross(&w).normalize();
        let v = w.cross(&u);
        
        let lower_left_corner = position - u * half_width - v * half_height - w;
        let horizontal = u * (2.0 * half_width);
        let vertical = v * (2.0 * half_height);

        Self {
            origin: position,
            lower_left_corner,
            horizontal,
            vertical,
            u, v, w,
            fov,
            aspect_ratio,
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let direction = self.lower_left_corner + 
                       self.horizontal * u + 
                       self.vertical * v - 
                       self.origin;
        Ray::new(self.origin, direction)
    }

    fn update(&mut self, position: Vec3, lookat: Vec3, vup: Vec3) {
        let theta = self.fov * PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = self.aspect_ratio * half_height;
        
        self.w = (position - lookat).normalize();
        self.u = vup.cross(&self.w).normalize();
        self.v = self.w.cross(&self.u);
        
        self.origin = position;
        self.lower_left_corner = position - self.u * half_width - self.v * half_height - self.w;
        self.horizontal = self.u * (2.0 * half_width);
        self.vertical = self.v * (2.0 * half_height);
    }
}

struct FlyCam {
    position: Vec3,
    front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl FlyCam {
    fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        let mut cam = Self {
            position,
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(0.0, 0.0, 0.0),
            world_up: Vec3::new(0.0, 1.0, 0.0),
            yaw,
            pitch,
            speed: 5.0,
            sensitivity: 0.1,
        };
        cam.update_vectors();
        cam
    }

    fn update_vectors(&mut self) {
        let front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        );
        self.front = front.normalize();
        self.right = self.front.cross(&self.world_up).normalize();
        self.up = self.right.cross(&self.front).normalize();
    }

    fn process_keyboard(&mut self, keys: &[Key], delta_time: f32) {
        let velocity = self.speed * delta_time;
        
        if keys.contains(&Key::W) {
            self.position = self.position + self.front * velocity;
        }
        if keys.contains(&Key::S) {
            self.position = self.position - self.front * velocity;
        }
        if keys.contains(&Key::A) {
            self.position = self.position - self.right * velocity;
        }
        if keys.contains(&Key::D) {
            self.position = self.position + self.right * velocity;
        }
        if keys.contains(&Key::Q) {
            self.position = self.position - self.up * velocity;
        }
        if keys.contains(&Key::E) {
            self.position = self.position + self.up * velocity;
        }
    }

    fn process_mouse(&mut self, mut xoffset: f32, mut yoffset: f32) {
        xoffset *= self.sensitivity;
        yoffset *= self.sensitivity;

        self.yaw += xoffset;
        self.pitch += yoffset;

        if self.pitch > 89.0 {
            self.pitch = 89.0;
        }
        if self.pitch < -89.0 {
            self.pitch = -89.0;
        }

        self.update_vectors();
    }

    fn get_view_direction(&self) -> Vec3 {
        self.front
    }

    fn get_lookat(&self) -> Vec3 {
        self.position + self.front
    }
}

#[derive(Clone)]
struct Texture {
    width: usize,
    height: usize,
    data: Vec<Vec3>,
}

impl Texture {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![Vec3::new(0.0, 0.0, 0.0); width * height],
        }
    }

    fn from_file(path: &str) -> Result<Self, String> {
        if let Ok(img) = ImageReader::open(Path::new(path)) {
            if let Ok(img) = img.decode() {
                let rgb_img = img.to_rgb8();
                let (width, height) = rgb_img.dimensions();
                
                let mut texture = Texture::new(width as usize, height as usize);
                
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        let index = (y as usize) * texture.width + (x as usize);
                        texture.data[index] = Vec3::new(
                            pixel[0] as f32 / 255.0,
                            pixel[1] as f32 / 255.0,
                            pixel[2] as f32 / 255.0,
                        );
                    }
                }
                
                return Ok(texture);
            }
        }
        
        // Fallback texture
        let mut texture = Texture::new(2, 2);
        texture.data[0] = Vec3::new(0.8, 0.8, 0.8);
        texture.data[1] = Vec3::new(0.2, 0.2, 0.2);
        texture.data[2] = Vec3::new(0.2, 0.2, 0.2);
        texture.data[3] = Vec3::new(0.8, 0.8, 0.8);
        Ok(texture)
    }

    fn sample(&self, u: f32, v: f32) -> Vec3 {
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        let index = y * self.width + x;
        self.data[index]
    }
}

#[derive(Clone)]
enum MaterialType {
    Diffuse,
    Metal,
    Dielectric,
    Emissive,
}

#[derive(Clone)]
struct Material {
    material_type: MaterialType,
    albedo: Vec3,
    albedo_texture: Option<Texture>,
    roughness: f32,
    refractive_index: f32,
    emission: Vec3,
}

impl Material {
    fn new_diffuse(albedo: Vec3) -> Self {
        Self {
            material_type: MaterialType::Diffuse,
            albedo,
            albedo_texture: None,
            roughness: 0.0,
            refractive_index: 1.0,
            emission: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn new_metal(albedo: Vec3, roughness: f32) -> Self {
        Self {
            material_type: MaterialType::Metal,
            albedo,
            albedo_texture: None,
            roughness: roughness.clamp(0.0, 1.0),
            refractive_index: 1.0,
            emission: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn new_dielectric(refractive_index: f32) -> Self {
        Self {
            material_type: MaterialType::Dielectric,
            albedo: Vec3::new(1.0, 1.0, 1.0),
            albedo_texture: None,
            roughness: 0.0,
            refractive_index,
            emission: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn new_emissive(color: Vec3, intensity: f32) -> Self {
        Self {
            material_type: MaterialType::Emissive,
            albedo: color,
            albedo_texture: None,
            roughness: 0.0,
            refractive_index: 1.0,
            emission: color * intensity,
        }
    }

    fn with_texture(mut self, texture: Texture) -> Self {
        self.albedo_texture = Some(texture);
        self
    }

    fn get_albedo(&self, u: f32, v: f32) -> Vec3 {
        if let Some(texture) = &self.albedo_texture {
            texture.sample(u, v)
        } else {
            self.albedo
        }
    }
}

// TEXTURAS
impl Material {
    fn grass_block() -> Self {
        Material::new_diffuse(Vec3::new(0.4, 0.7, 0.2))
            .with_texture(Texture::from_file("textures/grass_block.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn dirt() -> Self {
        Material::new_diffuse(Vec3::new(0.5, 0.4, 0.3))
            .with_texture(Texture::from_file("textures/dirt.jpg").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn stone() -> Self {
        Material::new_diffuse(Vec3::new(0.6, 0.6, 0.6))
            .with_texture(Texture::from_file("textures/stone.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn cobblestone() -> Self {
        Material::new_diffuse(Vec3::new(0.5, 0.5, 0.5))
            .with_texture(Texture::from_file("textures/cobblestone.jpg").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn oak_wood() -> Self {
        Material::new_diffuse(Vec3::new(0.6, 0.5, 0.3))
            .with_texture(Texture::from_file("textures/oak_log.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn oak_leaves() -> Self {
        Material::new_diffuse(Vec3::new(0.2, 0.6, 0.2))
            .with_texture(Texture::from_file("textures/oak_leaves.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn water() -> Self {
        Material::new_dielectric(1.33)
            .with_texture(Texture::from_file("textures/water.jpg").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn glass() -> Self {
        Material::new_dielectric(1.5)
            .with_texture(Texture::from_file("textures/glass.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn iron_block() -> Self {
        Material::new_metal(Vec3::new(0.8, 0.8, 0.8), 0.1)
            .with_texture(Texture::from_file("textures/iron_block.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn gold_block() -> Self {
        Material::new_metal(Vec3::new(1.0, 0.8, 0.0), 0.05)
            .with_texture(Texture::from_file("textures/gold_block.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn diamond_block() -> Self {
        Material::new_dielectric(2.4)
            .with_texture(Texture::from_file("textures/diamond_block.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn glowstone() -> Self {
        Material::new_emissive(Vec3::new(1.0, 0.9, 0.6), 2.0)
            .with_texture(Texture::from_file("textures/glowstone.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn bedrock() -> Self {
        Material::new_diffuse(Vec3::new(0.2, 0.2, 0.2))
            .with_texture(Texture::from_file("textures/bedrock.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn obsidian() -> Self {
        Material::new_diffuse(Vec3::new(0.1, 0.05, 0.2))
            .with_texture(Texture::from_file("textures/obsidian.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn lava() -> Self {
        Material::new_emissive(Vec3::new(1.0, 0.3, 0.1), 3.0)
            .with_texture(Texture::from_file("textures/lava.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn sand() -> Self {
        Material::new_diffuse(Vec3::new(0.9, 0.8, 0.6))
            .with_texture(Texture::from_file("textures/sand.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn coal_ore() -> Self {
        Material::new_diffuse(Vec3::new(0.4, 0.4, 0.4))
            .with_texture(Texture::from_file("textures/coal_ore.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn iron_ore() -> Self {
        Material::new_diffuse(Vec3::new(0.7, 0.6, 0.5))
            .with_texture(Texture::from_file("textures/iron_ore.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }

    fn diamond_ore() -> Self {
        Material::new_diffuse(Vec3::new(0.5, 0.7, 0.8))
            .with_texture(Texture::from_file("textures/diamond_ore.png").unwrap_or_else(|_| Texture::new(1, 1)))
    }
}

struct HitRecord<'a> {
    t: f32,
    point: Vec3,
    normal: Vec3,
    material: &'a Material,
    u: f32,
    v: f32,
}

struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

impl Sphere {
    fn new(center: Vec3, radius: f32, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin.subtract(&self.center);
        let a = ray.direction.dot(&ray.direction);
        let b = oc.dot(&ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;

        if discriminant > 0.0 {
            let temp = (-b - discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                let point = ray.point_at_parameter(temp);
                let normal = point.subtract(&self.center).normalize();
                
                let phi = point.z.atan2(point.x);
                let theta = (point.y / self.radius).asin();
                let u = 1.0 - (phi + PI) / (2.0 * PI);
                let v = (theta + PI / 2.0) / PI;
                
                return Some(HitRecord {
                    t: temp,
                    point,
                    normal,
                    material: &self.material,
                    u,
                    v,
                });
            }

            let temp = (-b + discriminant.sqrt()) / a;
            if temp < t_max && temp > t_min {
                let point = ray.point_at_parameter(temp);
                let normal = point.subtract(&self.center).normalize();
                
                let phi = point.z.atan2(point.x);
                let theta = (point.y / self.radius).asin();
                let u = 1.0 - (phi + PI) / (2.0 * PI);
                let v = (theta + PI / 2.0) / PI;
                
                return Some(HitRecord {
                    t: temp,
                    point,
                    normal,
                    material: &self.material,
                    u,
                    v,
                });
            }
        }
        None
    }
}

struct Cube {
    min: Vec3,
    max: Vec3,
    material: Material,
}

impl Cube {
    fn new(min: Vec3, max: Vec3, material: Material) -> Self {
        Self { min, max, material }
    }

    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut t_near = f32::NEG_INFINITY;
        let mut t_far = f32::INFINITY;
        let mut hit_normal = Vec3::new(0.0, 0.0, 0.0);
        let mut hit_face = 0; // 0=X, 1=Y, 2=Z

        for i in 0..3 {
            let origin = match i {
                0 => ray.origin.x,
                1 => ray.origin.y,
                2 => ray.origin.z,
                _ => 0.0,
            };
            let direction = match i {
                0 => ray.direction.x,
                1 => ray.direction.y,
                2 => ray.direction.z,
                _ => 0.0,
            };
            let min = match i {
                0 => self.min.x,
                1 => self.min.y,
                2 => self.min.z,
                _ => 0.0,
            };
            let max = match i {
                0 => self.max.x,
                1 => self.max.y,
                2 => self.max.z,
                _ => 0.0,
            };

            if direction == 0.0 {
                if origin < min || origin > max {
                    return None;
                }
            } else {
                let t1 = (min - origin) / direction;
                let t2 = (max - origin) / direction;

                if t1 > t2 {
                    if t2 > t_near {
                        t_near = t2;
                        hit_normal = match i {
                            0 => Vec3::new(-1.0, 0.0, 0.0),
                            1 => Vec3::new(0.0, -1.0, 0.0),
                            2 => Vec3::new(0.0, 0.0, -1.0),
                            _ => Vec3::new(0.0, 0.0, 0.0),
                        };
                        hit_face = i;
                    }
                    if t1 < t_far {
                        t_far = t1;
                    }
                } else {
                    if t1 > t_near {
                        t_near = t1;
                        hit_normal = match i {
                            0 => Vec3::new(1.0, 0.0, 0.0),
                            1 => Vec3::new(0.0, 1.0, 0.0),
                            2 => Vec3::new(0.0, 0.0, 1.0),
                            _ => Vec3::new(0.0, 0.0, 0.0),
                        };
                        hit_face = i;
                    }
                    if t2 < t_far {
                        t_far = t2;
                    }
                }

                if t_near > t_far || t_far < 0.0 {
                    return None;
                }
            }
        }

        if t_near > t_min && t_near < t_max {
            let point = ray.point_at_parameter(t_near);
            
            // Mapeo UV 
            let (u, v) = match hit_face {
                0 => {
                    // Cara X (izquierda/derecha)
                    let u = (point.z - self.min.z) / (self.max.z - self.min.z);
                    let v = 1.0 - (point.y - self.min.y) / (self.max.y - self.min.y);
                    (u, v)
                },
                1 => {
                    // Cara Y (arriba/abajo)
                    let u = (point.x - self.min.x) / (self.max.x - self.min.x);
                    let v = (point.z - self.min.z) / (self.max.z - self.min.z);
                    (u, v)
                },
                2 => {
                    // Cara Z (frente/atrás)
                    let u = 1.0 - (point.x - self.min.x) / (self.max.x - self.min.x);
                    let v = 1.0 - (point.y - self.min.y) / (self.max.y - self.min.y);
                    (u, v)
                },
                _ => (0.0, 0.0),
            };
            
            let u = u.fract().abs();
            let v = v.fract().abs();
            
            return Some(HitRecord {
                t: t_near,
                point,
                normal: hit_normal,
                material: &self.material,
                u,
                v,
            });
        }

        None
    }
}

struct Light {
    position: Vec3,
    color: Vec3,
    intensity: f32,
}

impl Light {
    fn new(position: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }
}

fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - *n * 2.0 * v.dot(n)
}

fn refract(v: &Vec3, n: &Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let uv = v.normalize();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    
    if discriminant > 0.0 {
        Some((uv - *n * dt) * ni_over_nt - *n * discriminant.sqrt())
    } else {
        None
    }
}

fn schlick(cosine: f32, refractive_index: f32) -> f32 {
    let r0 = ((1.0 - refractive_index) / (1.0 + refractive_index)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn random_in_unit_sphere() -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let p = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if p.length() < 1.0 {
            return p;
        }
    }
}

trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.hit(ray, t_min, t_max)
    }
}

impl Hittable for Cube {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.hit(ray, t_min, t_max)
    }
}

fn color(ray: &Ray, objects: &[Arc<dyn Hittable>], lights: &[Light], depth: i32) -> Vec3 {
    if depth <= 0 {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let mut closest_t = f32::INFINITY;
    let mut hit_record: Option<HitRecord> = None;

    for object in objects {
        if let Some(record) = object.hit(ray, 0.001, closest_t) {
            closest_t = record.t;
            hit_record = Some(record);
            
            if closest_t < 0.1 {
                break;
            }
        }
    }

    if let Some(record) = hit_record {
        let emitted = record.material.emission;
        
        match record.material.material_type {
            MaterialType::Emissive => {
                emitted
            }
            
            MaterialType::Diffuse => {
                
                let mut result = Vec3::new(0.0, 0.0, 0.0);
                
                if let Some(light) = lights.first() {
                    let light_dir = light.position.subtract(&record.point).normalize();
                    let diffuse_intensity = record.normal.dot(&light_dir).max(0.0) * light.intensity;
                    let albedo = record.material.get_albedo(record.u, record.v);
                    result = albedo.multiply_vec(&light.color) * diffuse_intensity;
                }
                
                let ambient_intensity = 0.3; 
                let albedo = record.material.get_albedo(record.u, record.v);
                let ambient = albedo * ambient_intensity;
                
                emitted + result + ambient
            }
            
            MaterialType::Metal => {
                
                let reflected = reflect(&ray.direction.normalize(), &record.normal);
                let scattered = Ray::new(record.point, reflected);
                let albedo = record.material.get_albedo(record.u, record.v);
                emitted + color(&scattered, objects, lights, depth - 1).multiply_vec(&albedo) * 0.8
            }
            
            MaterialType::Dielectric => {
              
                let outward_normal: Vec3;
                let ni_over_nt: f32;
                
                if ray.direction.dot(&record.normal) > 0.0 {
                    outward_normal = record.normal * -1.0;
                    ni_over_nt = record.material.refractive_index;
                } else {
                    outward_normal = record.normal;
                    ni_over_nt = 1.0 / record.material.refractive_index;
                }
                
               
                if let Some(refracted) = refract(&ray.direction, &outward_normal, ni_over_nt) {
                    let scattered = Ray::new(record.point, refracted);
                    color(&scattered, objects, lights, depth - 1) * 0.95
                } else {
                    let reflected = reflect(&ray.direction, &record.normal);
                    let scattered = Ray::new(record.point, reflected);
                    color(&scattered, objects, lights, depth - 1) * 0.95
                }
            }
        }
    } else {
        
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        let sky_top = Vec3::new(0.5, 0.7, 1.0);
        let sky_bottom = Vec3::new(0.2, 0.4, 0.8);
        sky_bottom * (1.0 - t) + sky_top * t
    }
}

struct FPSCounter {
    frame_count: u32,
    last_time: Instant,
    fps: f32,
}

impl FPSCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            last_time: Instant::now(),
            fps: 0.0,
        }
    }
    
    fn update(&mut self) -> f32 {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_time).as_secs_f32();
        
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_time = now;
        }
        
        self.fps
    }
}

fn create_skyblock_island() -> Vec<Arc<dyn Hittable>> {
    let mut objects: Vec<Arc<dyn Hittable>> = Vec::new();
    
  
    
    // Capa de bedrock 
    for x in -2..3 {
        for z in -2..3 {
            objects.push(Arc::new(Cube::new(
                Vec3::new(x as f32, -3.0, z as f32),
                Vec3::new((x + 1) as f32, -2.0, (z + 1) as f32),
                Material::bedrock(),
            )));
        }
    }
    
    // Capas de dirt
    for y in -2..0 {
        for x in -2..3 {
            for z in -2..3 {
                objects.push(Arc::new(Cube::new(
                    Vec3::new(x as f32, y as f32, z as f32),
                    Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                    Material::dirt(),
                )));
            }
        }
    }

    for x in -2..3 {
        for z in -2..3 {
            objects.push(Arc::new(Cube::new(
                Vec3::new(x as f32, 0.0, z as f32),
                Vec3::new((x + 1) as f32, 1.0, (z + 1) as f32),
                Material::grass_block(),
            )));
        }
    }
    
 
    
    // Tronco del árbol
    for y in 1..5 {
        objects.push(Arc::new(Cube::new(
            Vec3::new(0.0, y as f32, 0.0),
            Vec3::new(1.0, (y + 1) as f32, 1.0),
            Material::oak_wood(),
        )));
    }
    
    // Hojas del árbol 
    for x in -1..3 {
        for y in 4..7 {
            for z in -1..3 {
        
                if y < 6 && x >= 0 && x < 2 && z >= 0 && z < 2 {
                    continue;
                }
                objects.push(Arc::new(Cube::new(
                    Vec3::new(x as f32, y as f32, z as f32),
                    Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                    Material::oak_leaves(),
                )));
            }
        }
    }
    
 
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.8, 0.0),
        Material::water(),
    )));
    
    
    // Bloque de hierro
    objects.push(Arc::new(Cube::new(
        Vec3::new(2.0, 1.0, 2.0),
        Vec3::new(3.0, 2.0, 3.0),
        Material::iron_block(),
    )));
    
    // Bloque de oro
    objects.push(Arc::new(Cube::new(
        Vec3::new(-2.0, 1.0, 2.0),
        Vec3::new(-1.0, 2.0, 3.0),
        Material::gold_block(),
    )));

    objects.push(Arc::new(Sphere::new(
        Vec3::new(1.5, 1.5, -1.5),
        0.4,
        Material::diamond_block(),
    )));
    
    // === ILUMINACIÓN ===
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(-1.0, 2.0, 1.0),
        Vec3::new(0.0, 3.0, 2.0),
        Material::glowstone(),
    )));
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(1.0, 1.0, -2.0),
        Vec3::new(2.0, 2.0, -1.0),
        Material::cobblestone(),
    )));
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(-2.0, 1.0, -1.0),
        Vec3::new(-1.0, 3.0, 0.0),
        Material::obsidian(),
    )));
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(2.0, 2.0, 0.0),
        Vec3::new(3.0, 3.0, 1.0),
        Material::glass(),
    )));

    objects.push(Arc::new(Cube::new(
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(2.0, 0.0, 2.0),
        Material::coal_ore(),
    )));
    

    objects.push(Arc::new(Cube::new(
        Vec3::new(-1.0, -2.0, 0.0),
        Vec3::new(0.0, -1.0, 1.0),
        Material::iron_ore(),
    )));
    
    objects.push(Arc::new(Sphere::new(
        Vec3::new(0.5, -2.5, -1.5),
        0.3,
        Material::diamond_ore(),
    )));
    


    objects.push(Arc::new(Sphere::new(
        Vec3::new(2.5, 0.2, -2.5),
        0.3,
        Material::lava(),
    )));
    
    objects
}

fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    
    let mut window = Window::new(
        "Minecraft Skyblock - Ray Tracer",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });


    let objects = create_skyblock_island();

    
    // Luz solar principal
    let sun_light = Light::new(
        Vec3::new(10.0, 15.0, 5.0),
        Vec3::new(1.0, 0.95, 0.8), // Luz cálida del sol
        1.5,
    );
    
    // Luz ambiental del cielo
    let sky_light = Light::new(
        Vec3::new(-5.0, 10.0, -5.0),
        Vec3::new(0.6, 0.8, 1.0), // Luz azul del cielo
        0.8,
    );
    
    // Luz de la glowstone
    let glow_light = Light::new(
        Vec3::new(-0.5, 2.5, 1.5),
        Vec3::new(1.0, 0.9, 0.6), // Luz cálida de glowstone
        1.0,
    );
    
    let lights = vec![sun_light, sky_light, glow_light];
    
    // Configuración de cámara
    let mut flycam = FlyCam::new(Vec3::new(-5.0, 5.0, 8.0), -30.0, -20.0);
    
    let aspect_ratio = WIDTH as f32 / HEIGHT as f32;
    let mut camera = Camera::new(
        flycam.position,
        flycam.get_lookat(),
        flycam.up,
        60.0,
        aspect_ratio,
    );

    let mut last_time = Instant::now();
    let mut last_mouse_pos = (WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0);
    let mut first_mouse = true;
    let mut fps_counter = FPSCounter::new();

    window.limit_update_rate(Some(Duration::from_millis(33))); 

    println!("Controles: WASD + QE + Mouse, ESC para salir");


    while window.is_open() && !window.is_key_down(Key::Escape) {
        let current_time = Instant::now();
        let delta_time = current_time.duration_since(last_time).as_secs_f32();
        last_time = current_time;

        let pressed_keys: Vec<Key> = [
            Key::W, Key::A, Key::S, Key::D, Key::Q, Key::E
        ].iter().filter(|&&k| window.is_key_down(k)).cloned().collect();
        
        flycam.process_keyboard(&pressed_keys, delta_time);
        
        if let Some((x, y)) = window.get_mouse_pos(minifb::MouseMode::Pass) {
            if !first_mouse {
                let xoffset = x - last_mouse_pos.0;
                let yoffset = last_mouse_pos.1 - y;
                flycam.process_mouse(xoffset as f32, yoffset as f32);
            }
            last_mouse_pos = (x, y);
            first_mouse = false;
        }
        
        camera.update(flycam.position, flycam.get_lookat(), flycam.up);
        
     
        let current_fps = fps_counter.fps;
        buffer.par_chunks_mut(WIDTH).enumerate().for_each(|(y, row)| {
        
            let skip = if current_fps < 10.0 { 4 } else if current_fps < 20.0 { 2 } else { 1 };
            
            for x in (0..WIDTH).step_by(skip) {
                let u = x as f32 / WIDTH as f32;
                let v = (HEIGHT - y) as f32 / HEIGHT as f32;
                
                let ray = camera.get_ray(u, v);

                let max_depth = if current_fps < 15.0 { 2 } else { 4 };
                let col = color(&ray, &objects, &lights, max_depth);
                
                let r = (col.x.min(1.0).max(0.0) * 255.0) as u32;
                let g = (col.y.min(1.0).max(0.0) * 255.0) as u32;
                let b = (col.z.min(1.0).max(0.0) * 255.0) as u32;
                let color_value = (r << 16) | (g << 8) | b;


                for dx in 0..skip.min(WIDTH - x) {
                    row[x + dx] = color_value;
                }
            }
        });

        let fps = fps_counter.update();
        if fps > 0.0 {
            window.set_title(&format!("Minecraft Skyblock  - FPS: {:.1}", fps));
        }

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}