# image_search/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image as PILImage
import io
import numpy as np
from .models import Image as ImageModel
from .ai_search_optimized import ai_search

@csrf_exempt
@require_http_methods(["POST"])
def upload_and_index_image(request):
    """Upload image locally and create searchable index"""
    try:
        uploaded_file = request.FILES.get("image")
        title = request.POST.get("title", "Untitled")
        description = request.POST.get("description", "")
        
        if not uploaded_file:
            return JsonResponse({"error": "No image provided"}, status=400)
        
        # Read file for vector extraction
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset pointer for Django save
        
        # Create PIL Image for vector extraction
        img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # Extract feature vector
        vector = ai_search.get_feature_vector(img)
        
        # Save to database
        image_obj = ImageModel.objects.create(
            title=title,
            description=description,
            image_file=uploaded_file,
            feature_vector=vector.tolist(),
            uploaded_by=request.user if request.user.is_authenticated else None
        )
        
        # ✅ FIX: Convert ALL IDs to string
        return JsonResponse({
            "status": "success",
            "image_id": str(image_obj.id),  # Convert ObjectId to string
            "image_url": request.build_absolute_uri(image_obj.image_file.url),
            "message": "Image uploaded and indexed successfully"
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug: print full error
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def search_similar_images(request):
    """Search for similar images using uploaded query image"""
    try:
        uploaded_file = request.FILES.get("image")
        threshold = float(request.POST.get("threshold", 0.7))
        limit = int(request.POST.get("limit", 10))
        
        if not uploaded_file:
            return JsonResponse({"error": "No image provided"}, status=400)
        
        # Open query image
        img = PILImage.open(uploaded_file).convert("RGB")
        
        # Search similar images
        results = ai_search.search_similar_images(
            img, 
            threshold=threshold, 
            limit=limit
        )
        
        # Build results with local URLs
        similar_images = []
        for img_obj, score in results:
            similar_images.append({
                "id": str(img_obj.id),  # ✅ Convert to string
                "title": img_obj.title,
                "description": img_obj.description,
                "image_url": request.build_absolute_uri(img_obj.image_file.url),
                "similarity_score": round(score * 100, 2),
                "uploaded_at": img_obj.uploaded_at.isoformat() if img_obj.uploaded_at else None
            })
        
        return JsonResponse({
            "status": "success",
            "query": "visual_search",
            "results_count": len(similar_images),
            "results": similar_images
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def find_duplicates(request):
    """Find duplicate images in database using cached vectors"""
    try:
        threshold = float(request.GET.get("threshold", 0.95))
        
        # Get all images with cached vectors
        all_images = list(ImageModel.objects.exclude(
            feature_vector__isnull=True
        ))
        
        duplicates = []
        processed = set()
        
        # Compare using cached vectors
        for i, img1 in enumerate(all_images):
            img1_id_str = str(img1.id)  # ✅ Convert once
            if img1_id_str in processed:
                continue
            
            query_vector = np.array(img1.feature_vector)
            similar = []
            
            for img2 in all_images[i+1:]:
                img2_id_str = str(img2.id)  # ✅ Convert once
                if img2_id_str in processed:
                    continue
                
                db_vector = np.array(img2.feature_vector)
                similarity = ai_search.calculate_similarity(query_vector, db_vector)
                
                if similarity >= threshold:
                    similar.append((img2, similarity))
            
            if similar:
                duplicate_group = {
                    "original": {
                        "id": str(img1.id),  # ✅ Convert to string
                        "title": img1.title,
                        "url": request.build_absolute_uri(img1.image_file.url)
                    },
                    "duplicates": []
                }
                
                for img, score in similar:
                    duplicate_group["duplicates"].append({
                        "id": str(img.id),  # ✅ Convert to string
                        "title": img.title,
                        "url": request.build_absolute_uri(img.image_file.url),
                        "similarity": round(score * 100, 2)
                    })
                
                duplicates.append(duplicate_group)
                processed.add(str(img1.id))
                processed.update(str(img.id) for img, _ in similar)
        
        return JsonResponse({
            "status": "success",
            "duplicates_count": len(duplicates),
            "duplicates": duplicates
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)