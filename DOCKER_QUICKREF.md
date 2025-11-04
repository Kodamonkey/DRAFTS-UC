# 🐳 Docker - Referencia Rápida

## ⚡ Comandos Esenciales

### Build (solo primera vez)
```bash
docker-compose build drafts-gpu    # GPU (10-15 min)
docker-compose build drafts-cpu    # CPU (5-8 min)
```

### Ejecutar Pipeline
```bash
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "archivo"
```

### Útiles
```bash
# Shell interactivo
docker-compose run --rm --entrypoint /bin/bash drafts-gpu

# Ayuda
docker-compose run --rm drafts-gpu --help

# Limpiar
docker-compose down --rmi all --volumes
```

---

## 📁 Estructura Requerida

```
DRAFTS-UC/
├── src/models/
│   ├── cent_resnet18.pth    ← Necesario
│   └── class_resnet18.pth   ← Necesario
├── Data/raw/
│   └── *.fits, *.fil        ← Tus datos
└── Results/                  ← Se crea automáticamente
```

---

## ✅ Checklist

- [ ] Docker Desktop corriendo (`docker ps` funciona)
- [ ] Modelos en `src/models/*.pth`
- [ ] Datos en `Data/raw/`
- [ ] Build completado

---

## 🚀 Ejemplo Completo

```bash
# 1. Build
docker-compose build drafts-gpu

# 2. Ejecutar
docker-compose run --rm drafts-gpu \
  --data-dir /app/Data/raw \
  --results-dir /app/Results \
  --target "2017-04-03-08_55_22_153_0006_t23.444" \
  --det-prob 0.3 \
  --class-prob 0.5

# 3. Ver resultados
ls Results/
```

---

## 🐛 Problemas Comunes

| Error | Solución |
|-------|----------|
| Docker daemon not running | Abre Docker Desktop |
| Modelos no encontrados | Verifica `src/models/*.pth` |
| Permission denied | `icacls Results /grant Everyone:F /T` |
| CUDA OOM | Usa `drafts-cpu` en lugar de `drafts-gpu` |

---

**Ver más:** README.md sección "Running with Docker"

