# from django.db import models
# from django.contrib.auth.models import User
# import json

# # class Image(models.Model):
# #     title = models.CharField(max_length=255)
# #     description = models.TextField(blank=True, null=True)
# #     feature_vector = models.JSONField(null=True, blank=True) 
# class Image(models.Model):
#     title = models.CharField(max_length=255)
#     description = models.TextField(blank=True, null=True)
#     supabase_path = models.TextField()
#     feature_vector = models.JSONField(null=True, blank=True)   # ← QUAN TRỌNG
#     uploaded_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
#     uploaded_at = models.DateTimeField(auto_now_add=True)
    
#     # Local storage (optional - có thể để trống nếu dùng Supabase)
#     image_file = models.ImageField(
#         upload_to='images/%Y/%m/%d/', 
#         blank=True, 
#         null=True
#     )
#     thumbnail = models.ImageField(
#         upload_to='thumbnails/%Y/%m/%d/', 
#         blank=True, 
#         null=True
#     )
    
#     # VECTOR EMBEDDING - Lưu dạng JSON để tương thích với SQLite
#     feature_vector = models.JSONField(
#         blank=True,
#         null=True,
#         help_text="AI feature vector for similarity search (list of floats)"
#     )
    
#     # Supabase storage path
#     supabase_path = models.CharField(
#         max_length=500, 
#         blank=True, 
#         null=True,
#         help_text="Path to image in Supabase storage"
#     )
    
#     uploaded_by = models.ForeignKey(
#         User, 
#         on_delete=models.CASCADE, 
#         related_name='images',
#         null=True,
#         blank=True
#     )
#     uploaded_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
#     tags = models.ManyToManyField('Tag', related_name='images', blank=True)
#     views_count = models.IntegerField(default=0)
    
#     class Meta:
#         ordering = ['-uploaded_at']
#         indexes = [
#             models.Index(fields=['-uploaded_at']),
#             models.Index(fields=['title']),
#         ]
    
#     def __str__(self):
#         return self.title
    
#     def get_public_url(self):
#         """Get public URL from Supabase"""
#         if self.supabase_path:
#             from .supabase_client import supabase_client
#             return supabase_client.get_public_url('images', self.supabase_path)
#         return None


# class Tag(models.Model):
#     name = models.CharField(max_length=50, unique=True)
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     def __str__(self):
#         return self.name


# class SearchHistory(models.Model):
#     user = models.ForeignKey(
#         User, 
#         on_delete=models.CASCADE, 
#         related_name='search_history'
#     )
#     query = models.CharField(max_length=255)
#     results_count = models.IntegerField(default=0)
#     searched_at = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         ordering = ['-searched_at']
#         verbose_name_plural = 'Search histories'
    
#     def __str__(self):
#         return f"{self.user.username} - {self.query}"
# image_search/models.py
from django.db import models
from django.contrib.auth.models import User


class Tag(models.Model):
    """Define Tag FIRST - before Image model"""
    name = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "tags"
    
    def __str__(self):
        return self.name


class Image(models.Model):
    """Image model - references Tag above"""
    id = models.AutoField(primary_key=True)  # Fix ObjectId issue
    title = models.CharField(max_length=255, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    
    # Lưu ảnh local
    image_file = models.ImageField(upload_to='images/%Y/%m/%d/')
    thumbnail = models.ImageField(upload_to='thumbnails/%Y/%m/%d/', blank=True, null=True)
    
    # AI feature vector
    feature_vector = models.JSONField(
        blank=True, 
        null=True, 
        help_text="AI feature vector for similarity search"
    )
    
    # Metadata
    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='images'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Tags & statistics
    tags = models.ManyToManyField(Tag, related_name='images', blank=True)  # Now Tag is defined
    views_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = "images"
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['-uploaded_at']),
            models.Index(fields=['title']),
        ]
    
    def __str__(self):
        return self.title or f"Image {self.id}"
    
    def get_image_url(self):
        """Get image URL"""
        if self.image_file:
            return self.image_file.url
        return None


class SearchHistory(models.Model):
    """Search history tracking"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_history')
    query = models.CharField(max_length=255)
    results_count = models.IntegerField(default=0)
    searched_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "search_history"
        ordering = ['-searched_at']
        verbose_name_plural = 'Search histories'
    
    def __str__(self):
        return f"{self.user.username} - {self.query}"