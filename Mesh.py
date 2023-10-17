import pymeshlab
import vedo
import os
from pathlib import Path
import json
import numpy as np

class Mesh:
    """ A class to represent a mesh.
        Provides methods to load (init), normalize, serialize and save a mesh.
        Also provides methods to convert the mesh to vedo and pymeshlab objects.
    """
    def __init__(self, file_path: str, meta_data_path: str = None):
        """ Initialize a Mesh object from a file path.

        Also loads normalized mesh if that was saved before, either in the same directory as the mesh file
        or in a different directory specified by the meta_data_path.

        Args:
            file_path (str): The absolute path to the (original) mesh file.
            meta_data_path (str): The absolute path to the meta data file (json and normalized mesh).
                Defaults to same directory as the (original) mesh file.

        Returns:
            Mesh: A new Mesh object loaded from the file path.
        """
        if(meta_data_path == None):
            meta_data_path = file_path.removesuffix(Path(file_path).name)

        self.file_path = file_path
        mesh = vedo.Mesh(file_path)
        self._original_vertices = mesh.points()
        self._original_faces = mesh.faces()

        #load normalized mesh if it exists
        normalized_file_path = self._normalized_file_path(output_dir=meta_data_path)
        if(os.path.isfile(normalized_file_path)):
            mesh = vedo.Mesh(normalized_file_path)
            self._normalized_vertices = mesh.points()
            self._normalized_faces = mesh.faces()
            self.normalized = True
        else:
            self._normalized_vertices = None
            self._normalized_faces = None
            self.normalized = False

    def serialize(self, original = False):
        ms = self.as_pymeshlab(original)
        measures = ms.get_geometric_measures()
        box = measures['bbox']
        bounding_box = {"x": box.dim_x(),
                        "y": box.dim_y(),
                        "z": box.dim_z(),
                        "diagonal": box.diagonal(),
                        #"min": box.min(), # <- these are numpy arrays, which are not serializable
                        #"max": box.max(), # plus they are not needed
                        }

        vertices, faces = self._get_internal_mesh(original)
        return {"name": Path(self.file_path).name,
                "class": self.get_shape_class(),
                "filepath": self.file_path,
                "file_extension": Path(self.file_path).suffix,
                "vertices": len(vertices),
                "faces": len(faces),
                "normalized": self.normalized and not original,
                "bounding_box": bounding_box,
                "barycenter": measures['barycenter'].tolist(),
                #all face types in the database are triangles, so we don't need to calculate this
                #"face_types": count_face_types(self.as_vedo()),
                }


    def get_shape_class(self):
        return self.file_path.replace("\\"+Path(self.file_path).name, "").split("\\")[-1]

    def normalize(self, target_num_vertices = 10000, acceptable_range = 0.2):
        """ Set normalization for the mesh.

        Set to fit a unit cube, and translate the barycenter to the origin.
        TODO: extend with normalization steps from assignment step 3.
        Normalizing multiple times will not change the mesh, only the first time will.

        Args:
            normalized (bool): If True, the mesh will be normalized.
                If False, the mesh will be returned to its original state.
        Returns:
            None
        """
        if self.normalized:
            return

        resized = False
        #Step 0. Remeshing. Do it first to reduce the number of vertices if necessary to speed up the rest of the process
        num_points = len(self._original_vertices)
        if(num_points > target_num_vertices * (1 + acceptable_range)):
            mesh = self.as_vedo()
            mesh.decimate(n=10000)
            pymesh = self.as_pymeshlab(data=[mesh.points(), mesh.faces()])
            resized = True
        else: 
            pymesh = self.as_pymeshlab()

        #Step 1. Translation. Let the barycenter coincide with the origin
        barycenter = pymesh.get_geometric_measures()['barycenter']
        pymesh.compute_matrix_from_translation(traslmethod = 'Set new Origin', neworigin = barycenter)


        #Step 2. Pose alignment using PCA
        pymesh.compute_matrix_by_principal_axis()

        #save normalized mesh in preparation for flipping (this uses as_vedo, and as_vedo uses the normalized mesh)
        self._normalized_vertices = pymesh.current_mesh().vertex_matrix()
        self._normalized_faces = pymesh.current_mesh().face_matrix()  
        #Step 3. Flipping (mirroring) the mesh if necessary. Moment test to determine the orientation of the mesh (most mass is upper left side)
        self._flip()

        #Step 4. Scale the mesh to fit in the unit cube
        pymesh.compute_matrix_from_scaling_or_normalization(uniformflag = True, unitflag = True, freeze = True)
                                                                                #freeze explicitly changes the vertices
                                                                                #not sure if this is necessary, find out later
        #save normalized mesh
        self._normalized_vertices = pymesh.current_mesh().vertex_matrix()
        self._normalized_faces = pymesh.current_mesh().face_matrix()
        self.normalized = True

        if(not resized):
            mesh = self.as_vedo()
            counter = 0
            while mesh.npoints < target_num_vertices * (1 - acceptable_range):
                mesh.subdivide(1, 4)
                counter +=1
                if counter > 2:
                    break
            #save normalized mesh
            self._normalized_vertices = pymesh.current_mesh().vertex_matrix()
            self._normalized_faces = pymesh.current_mesh().face_matrix()
            self.normalized = True
    
    def resample(self, target_num_vertices):
        """ Resample the mesh to have a certain number of vertices.

        Args:
            target_num_vertices (int): The number of vertices the mesh should have after resampling.
        """
        # pymesh = self.as_pymeshlab()
        # current_num_vertices = pymesh.current_mesh().vertex_number()
        # if(current_num_vertices < target_num_vertices):
        #     number_of_subdivisions = round(target_num_vertices / current_num_vertices)
        #     if(number_of_subdivisions <= 1):
        #         #subdivision would not change the mesh
        #         #it's close enough to the target number of vertices
        #         return
        #     pymesh.meshing_surface_subdivision_butterfly(iterations = number_of_subdivisions)
        # elif(current_num_vertices > target_num_vertices):
        #     pymesh.meshing_decimation_quadric_edge_collapse(targetfacenum = target_num_vertices)
        # else:
        #     return
        mesh_obj = self.as_vedo()
        counter = 0
        if mesh_obj.npoints > target_num_vertices:
            mesh_obj.decimate(n=10000)
        else:
            while mesh_obj.npoints < target_num_vertices:
                mesh_obj.subdivide(1, 4)
                counter +=1
                if counter > 2:
                    break
      

        self._normalized_vertices = mesh_obj.points()
        self._normalized_faces = mesh_obj.faces()
        
    def shape_characteristics(self):
        mesh = self.as_vedo()
        volume = mesh.volume()
        convex_hull = vedo.ConvexHull(mesh)

        convexity = volume / convex_hull.volume()
        print(f'convexity: {convexity}')

        area = mesh.area()
        print(f'area: {area}')

        #S3/(36pV2)
        compactness = pow(area, 3) / (36 * np.pi * pow(volume, 2))
        print(f'compactness: {compactness}')

        b = mesh.bounds()
        #dimensions of the bounding box
        length, width, height = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        rectangularity = volume / (length * width * height)
        print(f'rectangularity: {rectangularity}')

        # DIAGONAL
        #find the largest distance between any two surface points on the mesh
        #this is the diameter of the mesh
        # TODO: prove that the diameter is the largest distance between any two surface points on the mesh
        # is equal to the largest distance between any two vertices on the mesh
        # If we take the two surface points that is are not vertexes with the largest distance compared to any other pair
        # of surface points, then it is part of a polygonal surface (e.g. quad or triangle). It is clear that the normal plane of the 
        # vector between these points at any of these points, contains points that are further away from the other point than the point itself.
        # Since this third point forms a right (by definition of normal plane) triangle with the other two points, the line between the far point and the new point
        # is the hypothenuse of the triangle, and therefore longer than the line between the two original points.
        # if the polygonal surface has the same angle as such a normal, than the furthest such point to maximize the hypothenuse is a vertex, since 
        # a vertex is the furthest away before encountering the edge (from the vertex, you can only move closer to the point).
        # Thus, then the vertex must be farther away from the original point.
        # If the polygonal surface has a different angle than the normal, then there is a vertex that is behind the plane, and thus farther away from the original point,
        # since in this point, there is another normal plane with the vector between the two original points, but this plane is farther away from the original point.
        # therefore all points on this plane, are farther away from the other point than the original point itself, and the vertex is on this plane.
        # Therefore it is sufficient to only check the vertices, and since the vertices that are most apart occur on the convex hull, we only need to check the convex hull
        # Note: this must be a way to prove this more easily, but I don't know how yet.
        mesh_points = convex_hull.points()
        diagonal = 0
        for i in range(len(mesh_points)):
            for j in range(i+1, len(mesh_points)):
                distance = np.linalg.norm(mesh_points[i] - mesh_points[j])
                if(distance > diagonal):
                    diagonal = distance
        print(f"diagonal: {diagonal}")
        
        #TODO: eccentricity
        covariance(mesh.points())


        pass

    def save(self, output_dir = None):
        """ Save the meta data to a json file and possibly the normalized mesh.

        The json file will be saved in the same directory as the mesh file
            with the same name and the extension .json.
        The normalized mesh will be saved in the same directory as the mesh file
            with the same name and the extension _normalized.{file_extension}.

        Args:
            output_dir (str): The directory to save the meta data to.
                Defaults to same directory as the mesh file.
                Make sure the directory exists.
        """
        #create output dir if it doesn't exist
        if(output_dir != None and not os.path.isdir(output_dir)):
            os.mkdir(output_dir)

        #default output dir is the same as the mesh file
        if(output_dir == None):
            output_dir = self.file_path.removesuffix(Path(self.file_path).name)

        json_obj = json.dumps(self.serialize(original=True), indent=4)
        json_file_path = os.path.join(output_dir,
                                    Path(self.file_path).name.removesuffix(Path(self.file_path).suffix) + ".json")
        with open(json_file_path, "w") as outfile:
            outfile.write(json_obj)

        #also save the json for the normalized mesh
        if(self.normalized):
            json_obj = json.dumps(self.serialize(), indent=4)
            json_file_path = os.path.join(output_dir,
                                      Path(self.file_path).name.removesuffix(Path(self.file_path).suffix) + "_normalized.json")
            with open(json_file_path, "w") as outfile:
                outfile.write(json_obj)

        #save normalized mesh in meta data dir (or in the same dir as the mesh file by default)
        if(self.normalized):
            self.as_pymeshlab().save_current_mesh(self._normalized_file_path(output_dir=output_dir))


    ### PRIVATE METHODS ###
    def _normalized_file_path(self, output_dir = None):
        file_extension = Path(self.file_path).suffix
        if(output_dir == None):
            output_dir = self.file_path.removesuffix(Path(self.file_path).name)
        return os.path.join(output_dir, Path(self.file_path).name.removesuffix(file_extension) + "_normalized" + file_extension)

    def _get_internal_mesh(self, original = False):
        if(original or not self.normalized):
            vertices = self._original_vertices
            faces = self._original_faces
        else:
            vertices = self._normalized_vertices
            faces = self._normalized_faces
        return [vertices, faces]

    # Warning: as_vedo and as_pymeshlab are internal / private. This should be used internally only, not by the user.
    # however, sometimes it is useful to have access to a vedo mesh object, for example to show it in a custom way.
    # However, operations on the vedo mesh object or pymeshlab object will not be reflected in the Mesh object.
    def as_vedo(self, original = False):
        return vedo.Mesh(self._get_internal_mesh(original))

    def as_pymeshlab(self, original = False, data: list[list, list] = None):
        if data is None:
            vertices, faces = self._get_internal_mesh(original)
        else:
            vertices = data[0]
            faces = data[1]
        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertices, faces)
        ms.add_mesh(mesh)
        return ms

    # def _flip_slow(self):
    #     ''' Flips mesh according to the moment test formula from the slides.
    #     This is a slow method, and it is not used anymore.'''

    #     mesh = self.as_vedo()
    #     triangles = mesh.faces()
    #     #transform triangles list from list of vertex indices to list of coordinates
    #     triangles = [[mesh.points()[i] for i in triangle] for triangle in triangles]
    #     triangle_centers = [center_of_polygon(triangle) for triangle in triangles]
    #     #moment test to determine the orientation of the mesh (most mass is upper left side)
    #     #see Slides 4 Feature Extraction, slide 21
    #     x_mass, y_mass, z_mass = 0,0,0
    #     for center in triangle_centers:
    #         #get the sign of the x coordinate of the center
    #         x_mass += np.sign(center[0]) * pow(center[0], 2)
    #         y_mass += np.sign(center[1]) * pow(center[1], 2)
    #         z_mass += np.sign(center[2]) * pow(center[2], 2)
    #     #apply rotation to flip the mesh if it is lopsided compared to one of xy, yz, xz planes if necessary
    #     if(x_mass < 0):
    #         mesh.rotate_x(180)
    #     if(y_mass < 0):
    #         mesh.rotate_y(180)
    #     if(z_mass < 0):
    #         mesh.rotate_z(180)

    #     self._normalized_vertices = mesh.points()
    #     self._normalized_faces = mesh.faces()

    def _flip(self):
        ''' This is a different, much faster approach to flipping the mesh (using vedo library functions)
        It works by cutting the mesh with the xy, yz and xz planes, and then comparing the volumes of each side of the cut.
        Larger volumes should be on the left side of the cut, so if the volume on the right side is larger, the mesh is flipped.
        '''
        mesh = self.as_vedo()
        total_volume = mesh.volume()

        xy_plane = (0,0,1)
        yz_plane = (1,0,0)
        xz_plane = (0,1,0)
        planes = [xy_plane, yz_plane, xz_plane]
        plane_to_axis = {xy_plane: "z", yz_plane: "x", xz_plane: "y"}
        for plane in planes:
            #the cut_with_plane function mutates the vedo.Mesh object itself, so we need to make a copy
            copy_mesh = self.as_vedo()
            cut_mesh = copy_mesh.cut_with_plane(origin=(0,0,0), normal=plane)
            left_volume = cut_mesh.volume()
            right_volume = total_volume - left_volume
            if(right_volume > left_volume):
                mesh.mirror(axis=plane_to_axis[plane])

        #since this is a normalizing operation, we save it to the normalized mesh
        self._normalized_vertices = mesh.points()
        self._normalized_faces = mesh.faces()


''' Returns the center of a polygon
    Args:
        polygon (list): List of 3D vertices (x, y, z) of the polygon
    Returns:
        list: 3D coordinate (x, y, z) of the center of the polygon as a list
'''
def center_of_polygon(polygon):
    return [sum([p[0] for p in polygon])/len(polygon),
            sum([p[1] for p in polygon])/len(polygon),
            sum([p[2] for p in polygon])/len(polygon)]


def show(mesh: Mesh, before_after = False):
    """ Show the mesh in a vedo window.

    Args:
        mesh (Mesh): The mesh object to be shown
        before_after (bool): If True, the original mesh will be shown in red,
            the normalized mesh will be shown in blue.
            If False, only the normalized mesh will be shown in blue.

    Returns:
        None
    """
    after = mesh.as_vedo()
    after.color("blue")
    after.wireframe(True)
    if(before_after):
        before = mesh.as_vedo(original=True)
        before.color("red")
        before.wireframe(True)
        vedo.show(before, after, sharecam= False, N=2, axes=1).close()
    else:
        vedo.show(after, axes=1).close()

    # show before and after meshes next to each other
    # vedo.show(before, after, N=2, axes=1).close()



#note: i wrote this docstring to understand what this function does
def count_face_types(mesh_obj: vedo.Mesh):
    """ Get the distribution of face types of the mesh.

    Faces can be triangles, quads, or polygons with more than 4 vertices,
    and a mesh can have a mixture of these types or only one type.

    Args:
        mesh_obj (vedo.Mesh): The mesh object to be analyzed

    Returns:
        list<int,int>: A list of dict of face type (int) and its count (int).
            For example:
                [{"shape": 3, "count": 100}, {"shape": 4, "count": 200}, {"shape": 5, "count": 300}]
    """

    tmp_rst = {}
    for a_face in mesh_obj.faces():
        tmp_key = len(a_face)
        if tmp_key in tmp_rst.keys():
            tmp_rst[tmp_key] += 1
        else:
            tmp_rst[tmp_key] = 1
    rst = []
    for k, v in tmp_rst.items():
        rst.append({"shape": k,
                    "count": v})
    return rst

def covariance(points: np.ndarray):
    '''Compute the eigenvalues and eigenvectors from the covariance matrix of a set of 3D points
    Code taken from technical tips on the course website (under Computing PCA): 
        https://webspace.science.uu.nl/~telea001/MR/TechnicalTips

        Args:
            points (np.ndarray): 3xn matrix of 3D points (x, y, z) e.g. vedo.Points()

        Returns:
    '''
    # compute the covariance matrix for A 
    # see the documentation at 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
    # this function expects that each row of A represents a variable, 
    # and each column a single observation of all those variables
    points_cov = np.cov(points)  # 3x3 matrix

    # computes the eigenvalues and eigenvectors for the 
    # covariance matrix. See documentation at  
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    eigenvalues, eigenvectors = np.linalg.eig(points_cov)

    print("==> eigenvalues for (x, y, z)")
    print(eigenvalues)
    print("\n==> eigenvectors")
    print(eigenvectors)

    
    pass
