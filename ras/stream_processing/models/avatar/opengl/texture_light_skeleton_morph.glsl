#version 330

// based on moderngl_window's texture_light.glsl
// and extended by & adapted from three.js handling of skeleton/bones & (texture-based) handling morph targets

#if defined VERTEX_SHADER

#define USE_SKINNING
#define USE_MORPHTARGETS
// will be set to appropriate value upon shader compilation:
#define MORPHTARGETS_COUNT 0

//TODO define / undefine USE_SKINNING & USE_MORPHTARGETS during or before shader compilation:
//#define USE_SKINNING false
//#define USE_MORPHTARGETS false
//#if USE_SKINNING == false
//	#undef USE_SKINNING
//#endif
//#if USE_MORPHTARGETS == false
//	#undef USE_MORPHTARGETS
//#endif

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

out vec3 normal;
out vec2 uv;
out vec3 pos;

#ifdef USE_SKINNING
	in vec4 in_joints_0;
	in vec4 in_weights_0;
#endif

#ifdef USE_MORPHTARGETS
	uniform float morphTargetBaseInfluence;

	uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif

#ifdef USE_SKINNING
	uniform highp sampler2D boneTexture;

	mat4 getBoneMatrix( const in float i ) {

		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );

		return mat4( v1, v2, v3, v4 );

	}
#endif


void main() {

	vec3 objectNormal = vec3( in_normal );
    vec3 transformed = vec3( in_position );

#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( in_joints_0.x );
	mat4 boneMatY = getBoneMatrix( in_joints_0.y );
	mat4 boneMatZ = getBoneMatrix( in_joints_0.z );
	mat4 boneMatW = getBoneMatrix( in_joints_0.w );

	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += in_weights_0.x * boneMatX;
	skinMatrix += in_weights_0.y * boneMatY;
	skinMatrix += in_weights_0.z * boneMatZ;
	skinMatrix += in_weights_0.w * boneMatW;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
#endif

#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	//#ifdef MORPHTARGETS_TEXTURE
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif
#ifdef USE_SKINNING
	vec4 skinVertex = /* bindMatrix * */ vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * in_weights_0.x;
	skinned += boneMatY * skinVertex * in_weights_0.y;
	skinned += boneMatZ * skinVertex * in_weights_0.z;
	skinned += boneMatW * skinVertex * in_weights_0.w;
	transformed = ( /* bindMatrixInverse * */ skinned ).xyz;
#endif

    mat4 mv = m_cam * m_model; 
    vec4 p = mv * vec4( transformed, 1.0 );
	gl_Position = m_proj * p;
    mat3 m_normal = transpose(inverse(mat3(mv)));
    normal = m_normal * objectNormal;
    uv = in_texcoord_0;
    pos = p.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;

in vec3 normal;
in vec3 pos;
in vec2 uv;

void main()
{
    float l = dot(normalize(-pos), normalize(normal));
    vec4 color = texture(texture0, uv);
    fragColor = color * 0.25 + color * 0.75 * abs(l);
}

#endif
